#!/usr/bin/env python3
"""
Evaluate NF4-quantized / approximate-matmul models using lm-evaluation-harness
for standard prompt formatting and reproducible benchmark numbers.

Model loading and layer replacement are shared with run_benchmark_eval.py.
"""

from __future__ import annotations

import argparse
import sys
import os

import torch

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# Make sure the sibling module is importable when running from the scripts/ dir.
sys.path.insert(0, os.path.dirname(__file__))

from run_benchmark_eval import (
    load_nf4_model,
    dtype_map,
    _resolve_model_path,
)
from bitsandbytes.functional import set_nf4_ewm_lut_data

# lm-eval task names that correspond to the tasks in run_benchmark_eval.
LM_EVAL_TASKS: dict[str, str] = {
    "piqa":           "piqa",
    "siqa":           "social_iqa",
    "hellaswag":      "hellaswag",
    "winogrande":     "winogrande",
    "arc_easy":       "arc_easy",
    "arc_challenge":  "arc_challenge",
    "openbookqa":     "openbookqa",
    "commonsenseqa":  "commonsenseqa",
    "gsm8k_cot":          "gsm8k_cot",
    "humaneval":      "humaneval",
    "xsum":           "xsum",
    "wmt14_de_en":    "wmt14-de-en",
    "mmlu":           "mmlu",
    "mgsm_native_cot_en": "mgsm_native_cot_en",
    "mgsm_native_cot_de": "mgsm_native_cot_de",
    "mgsm_native_cot_es": "mgsm_native_cot_es",
    "mgsm_native_cot_fr": "mgsm_native_cot_fr",
    "mgsm_native_cot_ja": "mgsm_native_cot_ja",
    "mgsm_native_cot_zh": "mgsm_native_cot_zh",
    "mgsm_native_cot_ru": "mgsm_native_cot_ru",
    "mgsm_native_cot_bn": "mgsm_native_cot_bn",
    "mgsm_native_cot_sw": "mgsm_native_cot_sw",
    "mgsm_native_cot_te": "mgsm_native_cot_te",
    "mgsm_native_cot_th": "mgsm_native_cot_th",
}

# Per-task few-shot defaults from the Llama 3.1 evaluation table.
# Tasks not listed default to 0.
TASK_FEWSHOT: dict[str, int] = {
    "mmlu":                5,
    "gsm8k_cot":           8,
    "mgsm_native_cot_en":  8,
    "mgsm_native_cot_de":  8,
    "mgsm_native_cot_es":  8,
    "mgsm_native_cot_fr":  8,
    "mgsm_native_cot_ja":  8,
    "mgsm_native_cot_zh":  8,
    "mgsm_native_cot_ru":  8,
    "mgsm_native_cot_bn":  8,
    "mgsm_native_cot_sw":  8,
    "mgsm_native_cot_te":  8,
    "mgsm_native_cot_th":  8,
}

# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/llama-3_1-8b-instruct-nemo?version=2.0

ALL_TASKS = list(LM_EVAL_TASKS.keys())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate models via lm-evaluation-harness with custom layer variants."
    )
    parser.add_argument("--model", required=True,
                        help="Model name or local path.")
    parser.add_argument("--tasks", nargs="+",
                        choices=["all", *ALL_TASKS], default=["all"],
                        help="Tasks to run (default: all).")
    parser.add_argument("--num-fewshot", type=int, default=None,
                        help="Number of few-shot examples. Overrides per-task defaults when set.")
    parser.add_argument("--batch-size", default="auto",
                        help="Batch size passed to lm-eval (default: auto).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit examples per task (maps to lm-eval limit).")
    parser.add_argument("--load-dtype", choices=dtype_map.keys(), default="fp16",
                        help="Compute dtype for the model.")
    parser.add_argument(
        "--linear_layer",
        choices=[
            "Linear", "Linear4bit", "Linear4bitFakeQuantAct", "LinearNF4Compute",
            "LinearApproxFP32", "LinearApproxFP16",
            "LinearApproxFP8E4M3", "LinearApproxFP8E5M2", "LinearApproxBfloat16",
        ],
        default="Linear4bit",
        help="Type of linear layer to use.",
    )
    parser.add_argument(
        "--prim8-lut",
        choices=[
            "44R0", "55R0", "66R0", "77R0", "88R0", "99R0", "aaR0",
            "44R12", "55R12", "66R12", "77R12", "88R12", "99R12", "aaR12",
        ],
        default="aaR0",
        help="PRIM8 LUT variant for LinearApproxBfloat16.",
    )
    parser.add_argument("--output-path", default=None,
                        help="Optional path to write lm-eval JSON results.")
    parser.add_argument("--collect-stats", default=None, metavar="PATH",
                        help="Enable Linear4bit activation/weight stats collection and dump to PATH.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --- LUT setup (same as run_benchmark_eval) ---
    my_lut_256 = [1.0, 1.0, 0.5856893760119567, 0.5856893760119567, 0.36044934294192454, 0.2379464204868627, 0.10345854955312216, 0.0002454255115474395, -0.09275731925775, -0.21258522056358756, -0.2828921716640679, -0.41486347893929015, -0.6298858034284472, -0.6298858034284472, -1.0, -1.0, 1.0, 0.5856893760119567, 0.36044934294192454, 0.36044934294192454, 0.2379464204868627, 0.14409246451938135, 0.06691345227408212, 0.0002454255115474395, -0.05782855768018973, -0.14430502854452312, -0.21258522056358756, -0.2828921716640679, -0.41486347893929015, -0.41486347893929015, -0.6298858034284472, -1.0, 0.5856893760119567, 0.36044934294192454, 0.36044934294192454, 0.2379464204868627, 0.18619549755282502, 0.10345854955312216, 0.06691345227408212, 0.0002454255115474395, -0.05782855768018973, -0.09275731925775, -0.14430502854452312, -0.21258522056358756, -0.2828921716640679, -0.41486347893929015, -0.41486347893929015, -0.6298858034284472, 0.5856893760119567, 0.36044934294192454, 0.2379464204868627, 0.18619549755282502, 0.14409246451938135, 0.10345854955312216, 0.0399821557462507, 0.0002454255115474395, -0.03397266927048204, -0.09275731925775, -0.14430502854452312, -0.14430502854452312, -0.21258522056358756, -0.2828921716640679, -0.41486347893929015, -0.41486347893929015, 0.36044934294192454, 0.2379464204868627, 0.18619549755282502, 0.14409246451938135, 0.10345854955312216, 0.06691345227408212, 0.0399821557462507, 0.0002454255115474395, -0.03397266927048204, -0.05782855768018973, -0.09275731925775, -0.14430502854452312, -0.14430502854452312, -0.21258522056358756, -0.2828921716640679, -0.41486347893929015, 0.2379464204868627, 0.14409246451938135, 0.10345854955312216, 0.10345854955312216, 0.06691345227408212, 0.0399821557462507, 0.018906730858427517, 0.0002454255115474395, -0.01513044631792837, -0.03397266927048204, -0.05782855768018973, -0.09275731925775, -0.09275731925775, -0.14430502854452312, -0.14430502854452312, -0.21258522056358756, 0.10345854955312216, 0.06691345227408212, 0.06691345227408212, 0.0399821557462507, 0.0399821557462507, 0.018906730858427517, 0.018906730858427517, 0.0002454255115474395, -0.01513044631792837, -0.01513044631792837, -0.03397266927048204, -0.03397266927048204, -0.05782855768018973, -0.05782855768018973, -0.09275731925775, -0.09275731925775, 0.0002454255115474395, 0.0002454255115474395, 0.0002454255115474395, 0.0002454255115474395, 0.0002454255115474395, 0.0002454255115474395, 0.0002454255115474395, 0.0002454255115474395, 0.0002454255115474395, 0.0002454255115474395, 0.0002454255115474395, 0.0002454255115474395, 0.0002454255115474395, 0.0002454255115474395, 0.0002454255115474395, 0.0002454255115474395, -0.09275731925775, -0.05782855768018973, -0.05782855768018973, -0.03397266927048204, -0.03397266927048204, -0.01513044631792837, -0.01513044631792837, 0.0002454255115474395, 0.0002454255115474395, 0.018906730858427517, 0.018906730858427517, 0.0399821557462507, 0.0399821557462507, 0.06691345227408212, 0.06691345227408212, 0.10345854955312216, -0.21258522056358756, -0.14430502854452312, -0.09275731925775, -0.09275731925775, -0.05782855768018973, -0.03397266927048204, -0.01513044631792837, 0.0002454255115474395, 0.018906730858427517, 0.0399821557462507, 0.0399821557462507, 0.06691345227408212, 0.10345854955312216, 0.10345854955312216, 0.14409246451938135, 0.18619549755282502, -0.2828921716640679, -0.21258522056358756, -0.14430502854452312, -0.14430502854452312, -0.09275731925775, -0.05782855768018973, -0.03397266927048204, 0.0002454255115474395, 0.018906730858427517, 0.0399821557462507, 0.06691345227408212, 0.10345854955312216, 0.14409246451938135, 0.18619549755282502, 0.2379464204868627, 0.36044934294192454, -0.41486347893929015, -0.2828921716640679, -0.21258522056358756, -0.14430502854452312, -0.14430502854452312, -0.09275731925775, -0.03397266927048204, 0.0002454255115474395, 0.0399821557462507, 0.06691345227408212, 0.10345854955312216, 0.14409246451938135, 0.18619549755282502, 0.2379464204868627, 0.36044934294192454, 0.36044934294192454, -0.6298858034284472, -0.41486347893929015, -0.2828921716640679, -0.21258522056358756, -0.14430502854452312, -0.09275731925775, -0.05782855768018973, 0.0002454255115474395, 0.0399821557462507, 0.10345854955312216, 0.14409246451938135, 0.18619549755282502, 0.2379464204868627, 0.36044934294192454, 0.36044934294192454, 0.5856893760119567, -0.6298858034284472, -0.41486347893929015, -0.41486347893929015, -0.2828921716640679, -0.21258522056358756, -0.14430502854452312, -0.05782855768018973, 0.0002454255115474395, 0.06691345227408212, 0.10345854955312216, 0.18619549755282502, 0.2379464204868627, 0.36044934294192454, 0.36044934294192454, 0.5856893760119567, 0.5856893760119567, -1.0, -0.6298858034284472, -0.41486347893929015, -0.41486347893929015, -0.2828921716640679, -0.14430502854452312, -0.09275731925775, 0.0002454255115474395, 0.06691345227408212, 0.14409246451938135, 0.2379464204868627, 0.36044934294192454, 0.36044934294192454, 0.5856893760119567, 0.5856893760119567, 1.0, -1.0, -1.0, -0.6298858034284472, -0.41486347893929015, -0.41486347893929015, -0.21258522056358756, -0.09275731925775, 0.0002454255115474395, 0.10345854955312216, 0.18619549755282502, 0.36044934294192454, 0.36044934294192454, 0.5856893760119567, 0.5856893760119567, 1.0, 1.0]
    lut = torch.tensor(my_lut_256, device="cuda", dtype=torch.float32)
    set_nf4_ewm_lut_data(lut)

    # --- Model loading (same pipeline as run_benchmark_eval) ---
    model_path = _resolve_model_path(args.model)
    model, tokenizer = load_nf4_model(
        model_path, args.linear_layer, args.prim8_lut, dtype_map[args.load_dtype]
    )

    # --- Optional stats collection ---
    if args.collect_stats:
        from bitsandbytes.nn import Linear4bit
        stats_layers: dict[str, "Linear4bit"] = {}
        for name, module in model.named_modules():
            if isinstance(module, Linear4bit):
                module.enable_stats_collection()
                stats_layers[name] = module
        print(f"Stats collection enabled on {len(stats_layers)} Linear4bit layers.")

    # --- Wrap for lm-eval ---
    from lm_eval.models.huggingface import HFLM
    from lm_eval import simple_evaluate
    import lm_eval.utils as lm_utils

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    # --- Resolve task names ---
    if "all" in args.tasks:
        local_names = ALL_TASKS
    else:
        local_names = args.tasks

    lm_eval_task_names = [LM_EVAL_TASKS[t] for t in local_names]

    # --- Run evaluation (group by few-shot count so each group uses the right value) ---
    from collections import defaultdict
    groups: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for local, lm_name in zip(local_names, lm_eval_task_names):
        n = args.num_fewshot if args.num_fewshot is not None else TASK_FEWSHOT.get(local, 0)
        groups[n].append((local, lm_name))

    all_task_results: dict[str, dict] = {}
    for n_shots, task_pairs in sorted(groups.items()):
        group_lm_names = [lm_name for _, lm_name in task_pairs]
        r = simple_evaluate(
            model=lm,
            tasks=group_lm_names,
            num_fewshot=n_shots,
            limit=args.max_samples,
            log_samples=False,
            confirm_run_unsafe_code=True,
        )
        all_task_results.update(r.get("results", {}))
        if "results" in r:
            r["results"] = all_task_results
        results = r

    # --- Print summary ---
    print("\nSummary:")
    task_results = results.get("results", {})
    for local_name, lm_name in zip(local_names, lm_eval_task_names):
        r = task_results.get(lm_name, {})
        # lm-eval stores metrics as "metric,aggregation" keys, e.g. "acc,none"
        #acc = r.get("acc,none") or r.get("acc_norm,none") or r.get("exact_match,none")
        acc = r.get("acc_char,none") or r.get("acc_norm,none") or r.get("acc,none") or r.get("exact_match,none")
        if acc is not None:
            print(f"  {local_name:20s}: {acc * 100:.2f}%")
        else:
            # Fall back to printing whatever is there
            metrics = {k: v for k, v in r.items() if not k.startswith("alias")}
            print(f"  {local_name:20s}: {metrics}")

    # --- Dump activation/weight stats ---
    if args.collect_stats:
        import json
        layer_stats = {name: layer.get_stats() for name, layer in stats_layers.items()}

        # summary across all layers
        all_act_max      = [s["act_abs_max"]          for s in layer_stats.values()]
        all_ch_max       = [max(s["act_channel_abs_max"]) for s in layer_stats.values() if s["act_channel_abs_max"]]
        all_out_6        = [s["act_outlier_frac_6"]   for s in layer_stats.values()]
        all_out_100      = [s["act_outlier_frac_100"] for s in layer_stats.values()]
        all_out_1k       = [s["act_outlier_frac_1k"]  for s in layer_stats.values()]
        all_w_max        = [s["weight_absmax_max"]    for s in layer_stats.values() if s["weight_absmax_max"] is not None]

        print("\n--- Activation & Weight Stats Summary ---")
        print(f"  Layers collected       : {len(layer_stats)}")
        print(f"  act_abs_max  (max/mean): {max(all_act_max):.3e} / {sum(all_act_max)/len(all_act_max):.3e}")
        print(f"  channel_max  (max/mean): {max(all_ch_max):.3e} / {sum(all_ch_max)/len(all_ch_max):.3e}")
        print(f"  outlier>6    (max/mean): {max(all_out_6)*100:.4f}% / {sum(all_out_6)/len(all_out_6)*100:.4f}%")
        print(f"  outlier>100  (max/mean): {max(all_out_100)*100:.4f}% / {sum(all_out_100)/len(all_out_100)*100:.4f}%")
        print(f"  outlier>1k   (max/mean): {max(all_out_1k)*100:.4f}% / {sum(all_out_1k)/len(all_out_1k)*100:.4f}%")
        if all_w_max:
            print(f"  weight_absmax(max/mean): {max(all_w_max):.3e} / {sum(all_w_max)/len(all_w_max):.3e}")

        # top-5 layers by peak activation magnitude
        top5 = sorted(layer_stats.items(), key=lambda kv: kv[1]["act_abs_max"], reverse=True)[:5]
        print("\n  Top-5 layers by act_abs_max:")
        for name, s in top5:
            print(f"    {name:60s}  act_abs_max={s['act_abs_max']:.3e}  outlier>100={s['act_outlier_frac_100']*100:.4f}%")

        with open(args.collect_stats, "w") as f:
            json.dump(layer_stats, f, indent=2, default=str)
        print(f"\nPer-layer stats written to {args.collect_stats}")

    # --- Optional JSON dump ---
    if args.output_path:
        import json
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults written to {args.output_path}")


if __name__ == "__main__":
    main()
