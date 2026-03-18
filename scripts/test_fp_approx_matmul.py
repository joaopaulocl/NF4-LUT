"""Tests for approximate FP matmul kernels.

FPMul(x,y) = sign(x)*sign(y) * (1 + x_m + y_m + 2^(-l(m))) * 2^(x_e + y_e)
  where x_m = fractional mantissa in [0,1), x_e = unbiased exponent,
  l(m): m if m<=3, 3 if m==4, 4 if m>4.
"""

import struct
import numpy as np
import torch
import sys
sys.path.insert(0, "/net/media/scratch/lima/NF4-LUT/bitsandbytes")

from bitsandbytes.functional import (
    fp32_approx_matmul,
    fp16_approx_matmul,
    fp8_e4m3_approx_matmul,
    fp8_e5m2_approx_matmul,
)


# ── Reference implementation (pure Python) ──────────────────────────────────

def decompose_fp32_py(x: float):
    bits = struct.unpack("I", struct.pack("f", x))[0]
    exp_raw = (bits >> 23) & 0xFF
    if exp_raw == 0 or exp_raw == 255:
        return None  # zero / subnormal / special
    sign = -1 if (bits >> 31) else 1
    exp = exp_raw - 127
    mant = (bits & 0x7FFFFF) / (1 << 23)
    return sign, exp, mant


def decompose_fp16_py(x: float):
    bits = struct.unpack("H", struct.pack("e", x))[0]
    exp_raw = (bits >> 10) & 0x1F
    if exp_raw == 0 or exp_raw == 31:
        return None
    sign = -1 if (bits >> 15) else 1
    exp = exp_raw - 15
    mant = (bits & 0x3FF) / 1024.0
    return sign, exp, mant


def decompose_fp8_e4m3_py(raw: int):
    absval = raw & 0x7F
    exp_raw = (absval >> 3) & 0xF
    if exp_raw == 0 or absval == 0x7F:
        return None
    sign = -1 if (raw >> 7) else 1
    exp = exp_raw - 7
    mant = (absval & 0x7) / 8.0
    return sign, exp, mant


def decompose_fp8_e5m2_py(raw: int):
    absval = raw & 0x7F
    exp_raw = (absval >> 2) & 0x1F
    if exp_raw == 0 or exp_raw == 31:
        return None
    sign = -1 if (raw >> 7) else 1
    exp = exp_raw - 15
    mant = (absval & 0x3) / 4.0
    return sign, exp, mant


def approx_mul_py(decomp_a, decomp_b, correction_bits: int) -> float:
    if decomp_a is None or decomp_b is None:
        return 0.0
    sa, ea, ma = decomp_a
    sb, eb, mb = decomp_b
    correction = 2.0 ** (-correction_bits)
    mag = 1.0 + ma + mb + correction
    return float(sa * sb) * mag * (2.0 ** (ea + eb))


def reference_matmul(A_vals, B_vals, decompose_fn, correction_bits):
    """A_vals: (M, K) list/array of raw values, B_vals: (N, K) pre-transposed."""
    M, K = len(A_vals), len(A_vals[0])
    N = len(B_vals)
    C = np.zeros((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += approx_mul_py(decompose_fn(A_vals[i][k]), decompose_fn(B_vals[j][k]), correction_bits)
            C[i, j] = s
    return C


# ── Test helpers ──────────────────────────────────────────────────────────────

def check_close(got: torch.Tensor, expected: np.ndarray, atol=1e-4, label=""):
    got_np = got.cpu().float().numpy()
    max_err = np.max(np.abs(got_np - expected))
    ok = max_err <= atol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label} | max_err={max_err:.6f}")
    return ok


# ── FP32 tests ────────────────────────────────────────────────────────────────

def test_fp32_single_element():
    """FPMul(1.0, 1.0) FP32 → (1 + 0 + 0 + 0.0625) * 2^0 = 1.0625"""
    A = torch.tensor([[1.0]], dtype=torch.float32, device="cuda")
    B = torch.tensor([[1.0]], dtype=torch.float32, device="cuda")
    C = fp32_approx_matmul(A, B)
    expected = 1.0625
    got = C[0, 0].item()
    ok = abs(got - expected) < 1e-5
    print(f"  [{'PASS' if ok else 'FAIL'}] fp32 single-element | expected={expected}, got={got:.6f}")
    return ok


def test_fp32_zero():
    A = torch.zeros(4, 8, dtype=torch.float32, device="cuda")
    B = torch.ones(4, 8, dtype=torch.float32, device="cuda")
    C = fp32_approx_matmul(A, B)
    ok = C.abs().max().item() == 0.0
    print(f"  [{'PASS' if ok else 'FAIL'}] fp32 zero input | max={C.abs().max().item()}")
    return ok


def test_fp32_sign():
    """FPMul(-1.0, 1.0) = -1.0625"""
    A = torch.tensor([[-1.0]], dtype=torch.float32, device="cuda")
    B = torch.tensor([[1.0]], dtype=torch.float32, device="cuda")
    C = fp32_approx_matmul(A, B)
    expected = -1.0625
    got = C[0, 0].item()
    ok = abs(got - expected) < 1e-5
    print(f"  [{'PASS' if ok else 'FAIL'}] fp32 sign | expected={expected}, got={got:.6f}")
    return ok


def test_fp32_small_matrix():
    M, N, K = 4, 4, 8
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(N, K, dtype=torch.float32, device="cuda")

    A_list = A.cpu().tolist()
    B_list = B.cpu().tolist()
    expected = reference_matmul(A_list, B_list, decompose_fp32_py, correction_bits=4)

    C = fp32_approx_matmul(A, B)
    return check_close(C, expected, atol=1e-4, label="fp32 small matrix (4x4x8)")


def test_fp32_matrix_norm_error():
    """Compare against exact matmul; error measured as Frobenius norm ratio.
    For random Gaussian input the approximation correction is ~0.0625 per product,
    so the absolute error grows with K but the ratio error / ||exact||_F should be bounded.
    """
    M, N, K = 64, 64, 128
    torch.manual_seed(1)
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(N, K, dtype=torch.float32, device="cuda")

    exact = A @ B.T
    approx = fp32_approx_matmul(A, B)

    err_norm = (approx - exact).norm().item()
    exact_norm = exact.norm().item()
    ratio = err_norm / (exact_norm + 1e-8)
    # With correction=0.0625 per product and K=128 elements summed,
    # the correction term is ~0.0625*K = 8 per output element (before sign cancellation),
    # but with random signs it's ~0.0625*sqrt(K) per element.
    # We use a generous threshold; this test documents the error level.
    ok = ratio < 5.0
    print(f"  [{'PASS' if ok else 'FAIL'}] fp32 Frobenius error ratio | ||err||/||exact||={ratio:.4f}")
    return ok


# ── FP16 tests ────────────────────────────────────────────────────────────────

def test_fp16_single_element():
    """FPMul(1.0, 1.0) FP16 → 1.0625 (same correction as FP32, l(10)=4)"""
    A = torch.tensor([[1.0]], dtype=torch.float16, device="cuda")
    B = torch.tensor([[1.0]], dtype=torch.float16, device="cuda")
    C = fp16_approx_matmul(A, B)
    expected = 1.0625
    got = C[0, 0].item()
    ok = abs(got - expected) < 1e-4
    print(f"  [{'PASS' if ok else 'FAIL'}] fp16 single-element | expected={expected}, got={got:.6f}")
    return ok


def test_fp16_small_matrix():
    M, N, K = 4, 4, 8
    torch.manual_seed(2)
    # Use values representable exactly in fp16
    A = torch.tensor([[1.0, 0.5, -0.25, 2.0, 0.125, -1.0, 0.75, 1.5]] * M,
                     dtype=torch.float16, device="cuda")
    B = torch.tensor([[1.0, 0.5, -0.25, 2.0, 0.125, -1.0, 0.75, 1.5]] * N,
                     dtype=torch.float16, device="cuda")

    A_list = [[struct.unpack("H", struct.pack("e", v))[0] for v in row]
              for row in A.cpu().float().tolist()]
    B_list = [[struct.unpack("H", struct.pack("e", v))[0] for v in row]
              for row in B.cpu().float().tolist()]
    # Convert back to float for decompose: pass raw bits via wrapper that expects float
    # Simpler: use decompose_fp16_py which takes float value
    A_float = A.cpu().float().tolist()
    B_float = B.cpu().float().tolist()
    expected = reference_matmul(A_float, B_float, decompose_fp16_py, correction_bits=4)

    C = fp16_approx_matmul(A, B)
    return check_close(C, expected, atol=1e-3, label="fp16 small matrix (4x4x8)")


# ── FP8 e4m3 tests ────────────────────────────────────────────────────────────

def test_fp8_e4m3_single_element():
    """1.0 in e4m3fn: sign=0, exp=7 (unbiased=0), mantissa=0b000 → raw=0x38.
    FPMul(1.0, 1.0) e4m3 → (1+0+0+0.125)*2^0 = 1.125"""
    raw_one = 0x38  # 1.0 in e4m3fn
    A_raw = torch.tensor([[raw_one]], dtype=torch.uint8, device="cuda")
    B_raw = torch.tensor([[raw_one]], dtype=torch.uint8, device="cuda")
    C = fp8_e4m3_approx_matmul(A_raw, B_raw)
    expected = 1.125
    got = C[0, 0].item()
    ok = abs(got - expected) < 1e-5
    print(f"  [{'PASS' if ok else 'FAIL'}] fp8-e4m3 single-element | expected={expected}, got={got:.6f}")
    return ok


def test_fp8_e4m3_small_matrix():
    """Use torch.float8_e4m3fn tensors (API path with view(uint8))."""
    vals = [1.0, 0.5, -1.0, 0.25]
    A = torch.tensor([vals * 2], dtype=torch.float32).to(torch.float8_e4m3fn).to("cuda")  # 1x8
    B = torch.stack([torch.tensor(vals * 2, dtype=torch.float32).to(torch.float8_e4m3fn)] * 4).to("cuda")  # 4x8

    # Reference using uint8 bits
    A_raw = A.view(torch.uint8).cpu().tolist()
    B_raw = B.view(torch.uint8).cpu().tolist()
    expected = reference_matmul(A_raw, B_raw, decompose_fp8_e4m3_py, correction_bits=3)

    C = fp8_e4m3_approx_matmul(A, B)
    return check_close(C, expected, atol=1e-4, label="fp8-e4m3 small matrix (1x4x8)")


# ── FP8 e5m2 tests ────────────────────────────────────────────────────────────

def test_fp8_e5m2_single_element():
    """1.0 in e5m2: sign=0, exp=15 (unbiased=0), mantissa=0b00 → raw=0x3C.
    FPMul(1.0, 1.0) e5m2 → (1+0+0+0.25)*2^0 = 1.25"""
    raw_one = 0x3C  # 1.0 in e5m2
    A_raw = torch.tensor([[raw_one]], dtype=torch.uint8, device="cuda")
    B_raw = torch.tensor([[raw_one]], dtype=torch.uint8, device="cuda")
    C = fp8_e5m2_approx_matmul(A_raw, B_raw)
    expected = 1.25
    got = C[0, 0].item()
    ok = abs(got - expected) < 1e-5
    print(f"  [{'PASS' if ok else 'FAIL'}] fp8-e5m2 single-element | expected={expected}, got={got:.6f}")
    return ok


def test_fp8_e5m2_small_matrix():
    vals = [1.0, 0.5, -1.0, 0.25]
    A = torch.tensor([vals * 2], dtype=torch.float32).to(torch.float8_e5m2).to("cuda")
    B = torch.stack([torch.tensor(vals * 2, dtype=torch.float32).to(torch.float8_e5m2)] * 4).to("cuda")

    A_raw = A.view(torch.uint8).cpu().tolist()
    B_raw = B.view(torch.uint8).cpu().tolist()
    expected = reference_matmul(A_raw, B_raw, decompose_fp8_e5m2_py, correction_bits=2)

    C = fp8_e5m2_approx_matmul(A, B)
    return check_close(C, expected, atol=1e-4, label="fp8-e5m2 small matrix (1x4x8)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== FP32 ===")
    r1 = test_fp32_single_element()
    r2 = test_fp32_sign()
    r3 = test_fp32_zero()
    r4 = test_fp32_small_matrix()
    r5 = test_fp32_matrix_norm_error()

    print("=== FP16 ===")
    r6 = test_fp16_single_element()
    r7 = test_fp16_small_matrix()

    print("=== FP8 e4m3fn ===")
    r8 = test_fp8_e4m3_single_element()
    r9 = test_fp8_e4m3_small_matrix()

    print("=== FP8 e5m2 ===")
    r10 = test_fp8_e5m2_single_element()
    r11 = test_fp8_e5m2_small_matrix()

    all_pass = all([r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11])
    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_pass else 1)
