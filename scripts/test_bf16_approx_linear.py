"""
Test correctness of LinearApproxBfloat16 vs. standard nn.Linear.
Measures MSE, MAE, and relative error for single and batched inputs.

Usage:
    python scripts/test_bf16_approx_linear.py --prim8-lut aaR0
    python scripts/test_bf16_approx_linear.py --prim8-lut 55R12 --in-features 512 --out-features 256 --batch 64
    python scripts/test_bf16_approx_linear.py --prim8-lut aaR0 --no-bias --breakdown
    python scripts/test_bf16_approx_linear.py --prim8-lut aaR0 --scenario post-layernorm
    python scripts/test_bf16_approx_linear.py --prim8-lut aaR0 --scenario outlier
    python scripts/test_bf16_approx_linear.py --prim8-lut aaR0 --scenario ffn-gated
"""
import argparse
import torch
import torch.nn as nn
import bitsandbytes.functional as F
from bitsandbytes.nn import LinearApproxBfloat16

SCENARIOS = {
    "post-layernorm": {"lo": -5.0,  "hi":  5.0,  "desc": "Post-LayerNorm activations (typical transformer matmul input)"},
    "outlier":        {"lo": -50.0, "hi": 50.0,  "desc": "Outlier-heavy residual stream (before LayerNorm)"},
    "ffn-gated":      {"lo":  0.0,  "hi":  8.0,  "desc": "FFN gated path activations (after SiLU/GELU gate)"},
    "default":        {"lo": -2.0,  "hi":  2.0,  "desc": "Default uniform [-2, 2]"},
}

PRIM8_LUT_IDS = {
    "44R0": 0, "55R0": 1, "66R0": 2, "77R0": 3, "88R0": 4, "99R0": 5, "aaR0": 6,
    "44R12": 7, "55R12": 8, "66R12": 9, "77R12": 10, "88R12": 11, "99R12": 12, "aaR12": 13,
}


def metrics(approx: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    err = approx.float() - ref.float()
    mse = err.pow(2).mean().item()
    mae = err.abs().mean().item()
    rel = (err.abs() / (ref.float().abs() + 1e-12)).mean().item()
    print(f"  [{label}]  MSE={mse:.6e}  MAE={mae:.6e}  RelErr={rel:.6e}")


def make_layers(in_features, out_features, bias, device, seed):
    torch.manual_seed(seed)
    ref_layer = nn.Linear(in_features, out_features, bias=bias, device=device).bfloat16()

    approx_layer = LinearApproxBfloat16(in_features, out_features, bias=bias, device=device)
    # Share the same weight and bias so the only difference is the matmul kernel.
    approx_layer.load_state_dict(ref_layer.state_dict())
    approx_layer = approx_layer.to(device)

    return ref_layer, approx_layer


def run(args):
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"

    lut_id = PRIM8_LUT_IDS[args.prim8_lut]
    F.set_prim8_lut(lut_id)
    scenario = SCENARIOS[args.scenario]
    lo, hi   = scenario["lo"], scenario["hi"]

    print(f"PRIM8 LUT : {args.prim8_lut} (id={lut_id})")
    print(f"Layer     : in={args.in_features}  out={args.out_features}  bias={not args.no_bias}")
    print(f"Scenario  : {args.scenario} — {scenario['desc']}")
    print(f"Input     : batch={args.batch}  dtype=bfloat16  range=[{lo}, {hi}]")

    ref_layer, approx_layer = make_layers(
        args.in_features, args.out_features, not args.no_bias, device, args.seed
    )

    torch.manual_seed(args.seed + 1)
    x = (torch.rand(args.batch, args.in_features, device=device) * (hi - lo) + lo).bfloat16()

    with torch.no_grad():
        ref_out    = ref_layer(x)         # exact BF16 hardware matmul, cast back to BF16
        approx_out = approx_layer(x)      # approx PRIM8 LUT matmul, cast back to BF16

    print(f"\nRef output    shape={tuple(ref_out.shape)}  dtype={ref_out.dtype}")
    print(f"Approx output shape={tuple(approx_out.shape)}  dtype={approx_out.dtype}")

    print("\nError vs. nn.Linear (BF16 hardware matmul):")
    metrics(approx_out, ref_out, "approx vs ref")

    # Also compare against float32 reference (higher-precision baseline)
    with torch.no_grad():
        ref_f32_out = nn.functional.linear(
            x.float(),
            ref_layer.weight.float(),
            ref_layer.bias.float() if ref_layer.bias is not None else None,
        )
    print("\nError vs. nn.Linear (float32 reference):")
    metrics(approx_out, ref_f32_out, "approx vs f32")
    metrics(ref_out,    ref_f32_out, "bf16   vs f32")



    # --- 3-D input (sequence dimension) ---
    print("\n3-D input check (batch=4, seq=16):")
    torch.manual_seed(args.seed + 2)
    x3d = (torch.rand(4, 16, args.in_features, device=device) * (hi - lo) + lo).bfloat16()
    with torch.no_grad():
        out_ref3d    = ref_layer(x3d)
        out_approx3d = approx_layer(x3d)
    shape_ok = tuple(out_approx3d.shape) == (4, 16, args.out_features)
    print(f"  shape={tuple(out_approx3d.shape)}  shape_ok={shape_ok}")
    metrics(out_approx3d, out_ref3d, "approx vs ref")
    flat_approx = approx_out.float().flatten()
    flat_ref    = ref_out.float().flatten()

    abs_err   = (flat_approx - flat_ref).abs()
    max_err   = abs_err.max().item()
    worst_idx = abs_err.argmax().item()
    batch_idx = worst_idx // args.out_features
    out_idx   = worst_idx  % args.out_features
    print(f"\nMax absolute error (approx vs f32): {max_err:.6e}")
    print(f"  Worst output element: batch={batch_idx}, out_feature={out_idx}")
    #print(f"  Input  x[{batch_idx}]: {x[batch_idx].float().tolist()}")
    #print(f"  Weight w[{out_idx}]:   {ref_layer.weight[out_idx].float().tolist()}")

    # --- magnitude breakdown ---
    if args.breakdown:
        print("\nMSE breakdown by |ref| magnitude bucket:")
        buckets = [(0, 0.125), (0.125, 0.5), (0.5, 1.0), (1.0, 4.0), (4.0, 10.0), (10.0, 50.0), (50.0, float("inf"))]
        flat_approx = approx_out.float().flatten()
        flat_ref    = ref_out.float().flatten()
        r = flat_ref.abs()
        for lo, hi in buckets:
            mask = (r >= lo) & (r < hi)
            if mask.sum() == 0:
                continue
            mse = (flat_approx[mask] - flat_ref[mask]).pow(2).mean().item()
            print(f"  |ref| ∈ [{lo:.3f}, {hi:6.3f}): n={mask.sum():8d}  MSE={mse:.6e}")


def main():
    parser = argparse.ArgumentParser(description="Test LinearApproxBfloat16 vs nn.Linear")
    parser.add_argument("--prim8-lut", choices=list(PRIM8_LUT_IDS.keys()), default="aaR0")
    parser.add_argument("--in-features",  type=int, default=256)
    parser.add_argument("--out-features", type=int, default=256)
    parser.add_argument("--batch",        type=int, default=32)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--scenario",      choices=list(SCENARIOS.keys()), default="default",
                        help="Input activation scenario (default: default)")
    parser.add_argument("--no-bias",      action="store_true")
    parser.add_argument("--breakdown",    action="store_true",
                        help="Print MSE breakdown by output magnitude bucket")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
