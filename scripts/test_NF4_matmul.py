import torch
from bitsandbytes.functional import dequantize_4bit, dequantize_blockwise, dequantize_nf4, get_4bit_type, nf4_matmul, quantize_4bit, quantize_blockwise, quantize_nf4, QuantState
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(2)

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(-1, str(ROOT_DIR))

from nf4.constants import NF4_MAG, NF4_POS_MAG, NF4_MAG_dict
from nf4.luts import build_5bit_acc_lut, build_nf4_mul_lut
from nf4.products import flatten_products, pairwise_product_matrix
from nf4.luts import nf4_matmul as nf4_matmul_py

from bitsandbytes.nn import LinearNF4Compute

def test_exectime():
    M, N, K = 1024, 1024, 1024
    A = torch.rand((M, K//2), device=device)
    B = torch.rand((K//2, N), device=device)

    ## trad GEMM kernel
    C = A @ B

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    C = A @ B
    end_event.record()
    torch.cuda.synchronize()
    cuda_time = start_event.elapsed_time(end_event)
    print(f"CUDA matmul time: {cuda_time:.3f} ms")

    A = torch.randint(0, 16, (M, K//2), device=device, dtype=torch.uint8)
    B = torch.randint(0, 16, (K//2, N), device=device, dtype=torch.uint8)
    C = nf4_matmul(A, B)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    C = nf4_matmul(A, B)
    end_event.record()
    torch.cuda.synchronize()
    cuda_time = start_event.elapsed_time(end_event)
    print(f"CUDA NF4 matmul time: {cuda_time:.3f} ms")

def test_main():
    M, N, K = 10, 10, 10    
    products = pairwise_product_matrix()
    flattened, unique_vals, unique_pos = flatten_products(products)
    lut_5b = build_5bit_acc_lut(unique_pos) 
    mul_lut = build_nf4_mul_lut(NF4_POS_MAG, lut_5b)

    A = torch.randint(0, 16, (M, K//2), device=device, dtype=torch.uint8)
    B = torch.randint(0, 16, (K//2, N), device=device, dtype=torch.uint8)
    
    # Convert A and B to float using NF4_MAG_dict
    A_np = A.cpu().numpy()
    B_np = B.cpu().numpy()
    A_float_np = np.vectorize(NF4_MAG_dict.get, otypes=[float])(A_np)
    B_float_np = np.vectorize(NF4_MAG_dict.get, otypes=[float])(B_np)
    A_float = torch.tensor(A_float_np, device=device, dtype=torch.float32)
    B_float = torch.tensor(B_float_np, device=device, dtype=torch.float32)
    
    # Multiply A_float @ B_float
    C_float = A_float @ B_float
    
    print("Float matmul result:\n", C_float)


    C_ref = torch.tensor(nf4_matmul_py(A.cpu().numpy(), B.cpu().numpy(), mul_lut), dtype=torch.float32)
    C = nf4_matmul(A, B).cpu()
    print(C_ref.dtype, C.dtype)
    print("NF4 matmul result (4x4):\n", C_ref)
    print("Reference matmul result (4x4):\n", C)
    assert torch.allclose(C_ref, C, 1e-3), "NF4 matmul does not match reference!"


def test_main_quantized():
    M, N, K = 64, 64, 64
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)
    
    # Reference A @ B
    #ref = A @ B
    
    # Quantize A and B
    A_q, A_state = quantize_nf4(A, blocksize=K)
    B_q, B_state = quantize_nf4(B, blocksize=K)
    ref = A @ B
    print(ref.flatten()[:10]) 

    # Reshape to expected shapes for nf4_matmul
    A_q = A_q.view(M, K // 2)
    B_q = B_q.view(K // 2, N)
    
    # NF4 matmul
    result = nf4_matmul(A_q, B_q)
        
    # Scale by combined absmax
    absmax_combined = A_state.absmax * B_state.absmax
    result_scaled = result * absmax_combined

    print(result_scaled.flatten()[:10])
    
    print("Reference shape:", ref.shape)
    print("Result shape:", result_scaled.shape)
    print("Max diff:", (ref - result_scaled).abs().max().item())
    print("Reference norm:", ref.norm().item())
    print("Result norm:", result_scaled.norm().item())
    
    # Assert close
    assert torch.allclose(ref, result_scaled, atol=1e-1), "Quantized NF4 matmul does not match reference!"

def test_dequantize_comparison():
    # Create some float data
    A_float = torch.randn(10, 10, device=device, dtype=torch.float32)
    
    # Quantize
    A_q, A_state = quantize_nf4(A_float)
    
    # Dequantize using dequantize_nf4
    A_state.absmax = A_state.absmax / A_state.absmax  # Normalize absmax to 1.0 for comparison

    A_dequant = dequantize_nf4(A_q, A_state)
    print(A_dequant.flatten()[0:5])
    
    # Manually dequantize: unpack A_q to indices, then lookup in NF4_MAG_dict
    A_q_np = A_q.cpu().numpy()
    indices = []
    for byte in A_q_np.flatten():
        indices.append(byte >> 4)
        indices.append(byte & 0xF)
    indices = np.array(indices).reshape(A_float.shape)
    A_manual_dequant_np = np.vectorize(NF4_MAG_dict.get, otypes=[float])(indices)
    A_manual_dequant = torch.tensor(A_manual_dequant_np, device=device, dtype=torch.float32)
    print(A_manual_dequant.flatten()[0:5])
    # Compare
    print("Max diff in dequantization:", (A_dequant - A_manual_dequant).abs().max().item())
    assert torch.allclose(A_dequant, A_manual_dequant, atol=1e-6), "Dequantize functions do not match!"
    print("Dequantize functions match!")

def test_main_matmul():
    M, N, K = 64, 64, 64    
    products = pairwise_product_matrix()
    flattened, unique_vals, unique_pos = flatten_products(products)
    lut_5b = build_5bit_acc_lut(unique_pos) 
    mul_lut = build_nf4_mul_lut(NF4_POS_MAG, lut_5b)

    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)

    qa, SA = quantize_4bit(A, blocksize=64, quant_type="nf4")
    qb, SB = quantize_4bit(B, blocksize=64, quant_type="nf4")

    
    A_float = dequantize_4bit(qa, SA, blocksize=64, quant_type="nf4")
    B_float = dequantize_4bit(qb, SB, blocksize=64, quant_type="nf4")


    # Multiply A_float @ B_float
    C_float = A_float @ B_float.T
    
    print("Float matmul result:\n", C_float)

    qa = qa.view(M, K // 2)
    qb = qb.view(K // 2, N)
    
    C = nf4_matmul(qa, qb)
    a_absmax = SA.absmax.reshape(SA.absmax.shape[0], 1)
    b_absmax = SB.absmax.reshape(1, SB.absmax.shape[0])
    C = C * (a_absmax * b_absmax)
    print("Reference matmul result (4x4):\n", C)

    print("Reference norm:", C_float.norm().item())
    print("Output norm:", C.norm().item())
    print("Max diff:", (C_float - C).abs().max().item())
    
    assert torch.allclose(C_float, C, 1e-3), "NF4 matmul does not match reference!"
    assert abs(C_float.norm().item() - C.norm().item()) < 1.0, "Norms differ too much"
    test_linear_nf4_compute(A_float, B_float)


def test_4bit_quant():
    A1 = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    qa, SA = quantize_4bit(A1, blocksize=64, quant_type="nf4")
    print(qa.shape, qa.dtype)

    A2 = torch.zeros(1024, 1024, device=device, dtype=torch.float32)
    SA.dtype = A2.dtype
    dequantize_4bit(qa, SA, blocksize=64, quant_type="nf4", out = A2)

    err = (A1 - A2).abs().float()

    print(A1[0:5,0:5], A2[0:5,0:5])

    relerr = (err / (A1.abs().float() + 1e-8)).mean()
    err = err.mean()
    print(f"4bit NF4 quant/dequant mean abs error: {err.item():.6f}, mean rel error: {relerr.item():.6f}")
    
    

def test_linear_nf4_compute(A=None, B = None):
    # Test LinearNF4Compute layer
    layer = LinearNF4Compute(64, 64, bias=False, blocksize= 64, device=device)
    
    # Create random weights
    W = torch.randn(64, 64, dtype=torch.float32, device=device) if B == None else B
    
    # Load weights
    layer.load_state_dict({'weight': W})
    
    # Create input
    x = torch.randn(64, 64, device=device, dtype=torch.float32) if A != None else A
    
    # Reference: x @ W.T
    ref = x @ W.T
    
    # Forward pass
    with torch.no_grad():
        out = layer(x)
    print(out.flatten()[:10])    
    print(ref.flatten()[:10])    
    
    print("Reference norm:", ref.norm().item())
    print("Output norm:", out.norm().item())
    print("Max diff:", (ref - out).abs().max().item())
    
    # Check if norms are close (since NF4 is approximate)
    assert abs(ref.norm().item() - out.norm().item()) < 1.0, "Norms differ too much"

if __name__ == "__main__":
    #test_main()  
    #test_dequantize_comparison()
    #test_main_quantized()
    #test_main_dequantize() 
    #test_exectime()
    #test_4bit_quant()
    test_main_matmul()
    #test_linear_nf4_compute()