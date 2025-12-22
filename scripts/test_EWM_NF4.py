import torch
import bitsandbytes as bnb

# Make sure CUDA is available
assert torch.cuda.is_available(), "CUDA is required for NF4 quantization"

device = "cuda"
torch.manual_seed(42)

# Two example tensors
A = torch.randn(128, 64, device=device, dtype=torch.float16)
B = torch.randn(128, 64, device=device, dtype=torch.float16)
out = torch.zeros((128, 64), device=device, dtype=torch.float16)

# Quantize to NF4
A_q, A_state = bnb.functional.quantize_4bit(
    A,
    quant_type="nf4",        # NormalFloat4
    compress_statistics=True
)

B_q, B_state = bnb.functional.quantize_4bit(
    B,
    quant_type="nf4",
    compress_statistics=True
)

print("A_q dtype:", A_q.dtype)
print("B_q dtype:", B_q.dtype)
print("A_state:", A_state.dtype)
print("B_state:", B_state.dtype)

 
bnb.functional.multiply_nf4(
    A_q, B_q, 
    quant_state=A_state,
    out=out
)
print("Output dtype:", out)

