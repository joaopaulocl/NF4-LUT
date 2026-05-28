"""
Test correctness of approximate BF16 element-wise multiply kernels.
Measures MSE and MAE vs. exact BF16 multiply (torch reference).

Usage:
    python scripts/test_bf16_approx_ewmul.py --prim8-lut aaR0
    python scripts/test_bf16_approx_ewmul.py --prim8-lut 55R12 --size 1048576 --seed 7
"""
import argparse
import torch
import bitsandbytes.functional as F

# PRIM8 LUT name → id (must match set_prim8_lut() in ops.cu)
PRIM8_LUT_IDS = {
    "44R0": 0, "55R0": 1, "66R0": 2, "77R0": 3, "88R0": 4, "99R0": 5, "aaR0": 6,
    "44R12": 7, "55R12": 8, "66R12": 9, "77R12": 10, "88R12": 11, "99R12": 12, "aaR12": 13,
}
SCENARIOS = {
    "post-layernorm": {"lo": -5.0,  "hi":  5.0,  "desc": "Post-LayerNorm activations (typical transformer matmul input)"},
    "outlier":        {"lo": -50.0, "hi": 50.0,  "desc": "Outlier-heavy residual stream (before LayerNorm)"},
    "ffn-gated":      {"lo":  0.0,  "hi":  8.0,  "desc": "FFN gated path activations (after SiLU/GELU gate)"},
    "default":        {"lo": -2.0,  "hi":  2.0,  "desc": "Default uniform [-2, 2]"},
}

def metrics(approx: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    err = (approx.float() - ref.float())
    mse = err.pow(2).mean().item()
    mae = err.abs().mean().item()
    rel = (err.abs() / (ref.float().abs() + 1e-12)).mean().item()
    print(f"  [{label}]  MSE={mse:.6e}  MAE={mae:.6e}  RelErr={rel:.6e}")


def run(args):
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    torch.manual_seed(args.seed)

    # Upload selected LUT to GPU
    lut_id = PRIM8_LUT_IDS[args.prim8_lut]
    F.set_prim8_lut(lut_id)
    print(f"PRIM8 LUT: {args.prim8_lut} (id={lut_id})")
    scenario = SCENARIOS[args.scenario]
    lo, hi   = scenario["lo"], scenario["hi"]

    print(f"Scenario  : {args.scenario} — {scenario['desc']}")

    # Random BF16 inputs — uniform in [-2, 2] to cover mixed signs and exponents
    A = (torch.rand(args.size, device=device) * (hi - lo) + lo).bfloat16()
    B = (torch.rand(args.size, device=device) * (hi - lo) + lo).bfloat16()

    print(A.max(), B.max())

    # Reference: exact BF16 multiply, cast to float32 for comparison
    ref_bf16 = (A * B)                         # BF16 result (hardware-exact on GPU)
    ref_f32  = ref_bf16.float()
    print(ref_bf16)
    print(ref_f32)
    print(f"Reference: exact BF16 multiply → float32")

    # --- approx variant (output: float32) ---
    approx_f32 = F.bf16_approx_ewmul(A, B)
    print("\nApprox (sa-decomposition + LUT, output float32):")
    metrics(approx_f32, ref_f32, "vs ref_f32")

    # --- faithful variant (output: BF16) ---
    faithful_bf16 = F.bf16_approx_ewmul_faithful(A, B)
    print(faithful_bf16)
    print("\nFaithful (bit-level: XOR signs, sum exponents, LUT mantissa, output BF16):")
    metrics(faithful_bf16.float(), ref_f32, "vs ref_f32")
    metrics(faithful_bf16.float(), ref_bf16.float(), "vs ref_bf16")

    
    max_id = (faithful_bf16 - ref_f32).abs().argmax()
    print("A*B = approx and ref:", A[max_id], B[max_id], faithful_bf16[max_id], ref_f32[max_id])
    # --- special value sanity checks ---
    print("\nSpecial value checks (faithful):")
    zero  = torch.zeros(1, device=device, dtype=torch.bfloat16)
    one   = torch.ones(1, device=device, dtype=torch.bfloat16)
    neg   = torch.tensor([-1.0], device=device, dtype=torch.bfloat16)
    inf_t = torch.tensor([float("inf")], device=device, dtype=torch.bfloat16)

    checks = [
        ("0 × 1",    zero,  one,  0.0),
        ("1 × 1",    one,   one,  1.0),
        ("(-1) × 1", neg,   one, -1.0),
        ("1 × (-1)", one,   neg, -1.0),
        ("(-1)×(-1)",neg,   neg,  1.0),
    ]
    for name, a, b, expected in checks:
        got = F.bf16_approx_ewmul_faithful(a, b).item()
        status = "OK" if abs(got - expected) < 1e-3 else f"FAIL (expected {expected})"
        print(f"  {name:12s} = {got:+.4f}  {status}")

    inf_result = F.bf16_approx_ewmul_faithful(inf_t, one).item()
    print(f"  {'inf × 1':12s} = {inf_result}  {'OK' if torch.isinf(torch.tensor(inf_result)) else 'FAIL'}")

    # --- per-exponent breakdown (faithful) ---
    if args.breakdown:
        print("\nMSE breakdown by |ref| magnitude bucket (faithful):")
        buckets = [(0, 0.125), (0.125, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 10.0), (10.0, 40.0), (40.0, float("inf"))]
        r = ref_bf16.abs()
        f = faithful_bf16.float()
        for lo, hi in buckets:
            mask = (r >= lo) & (r < hi)
            if mask.sum() == 0:
                continue
            mse = (f[mask] - ref_bf16[mask]).pow(2).mean().item()
            print(f"  |ref| ∈ [{lo:.3f}, {hi:.3f}): n={mask.sum():8d}  MSE={mse:.6e}")


def main():
    parser = argparse.ArgumentParser(description="Test BF16 approximate element-wise multiply")
    parser.add_argument(
        "--prim8-lut",
        choices=list(PRIM8_LUT_IDS.keys()),
        default="aaR0",
        help="PRIM8 LUT variant to use (default: aaR0)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1 << 20,
        help="Number of elements (default: 1M)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument("--scenario",      choices=list(SCENARIOS.keys()), default="default",
                        help="Input activation scenario (default: default)")
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Print MSE breakdown by magnitude bucket",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
