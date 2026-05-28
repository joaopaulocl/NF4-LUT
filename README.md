# NF4 LUT tools

Run the NF4 LUT analysis from the repository root:

```
python -m scripts.run_nf4_luts
```

Alternatively you can execute the script file directly:

```
python scripts/run_nf4_luts.py
```

The script configures `sys.path` so it can import the `nf4` helpers without requiring `PYTHONPATH`.

## LLAMA NF4 inference

You can run LLAMA inference using bitsandbytes NF4 quantization with:

```
python scripts/run_llama_nf4_inference.py --model <model-or-path> --prompt "tell me about NF4"
```

The script exposes a few helpful flags:

- `--use-chat-template` applies the tokenizer chat template when available.
- `--do-sample` toggles sampling (otherwise greedy decoding is used).
- `--max-new-tokens`, `--temperature`, and `--top-p` expose the usual `generate` controls.

## Downloading Hugging Face models

To download models locally (for later NF4 inference), run:

```
python scripts/download_hf_models.py meta-llama/Llama-2-7b-hf --cache-dir ./models
python scripts/download_hf_models.py meta-llama/Llama-3.1-8B-Instruct  --cache-dir ./models

```

You can pass multiple model IDs, add `--revision` to pin a commit/tag, and filter files with `--allow-pattern` / `--ignore-pattern`.


## WikiText-2 perplexity

Estimate perplexity on WikiText-2 with your NF4 checkpoint via:

```
python scripts/run_wikitext2_perplexity.py --model <model-or-path>
```

Flags worth tweaking:

- `--max-length` / `--stride` control the evaluation sliding window.
- `--max-texts` can cap the number of documents for quicker runs.


## Commonsense benchmarks

Evaluate PIQA, SIQA, HellaSwag, WinoGrande, ARC (easy/challenge), OpenBookQA, and CommonsenseQA with:

```
python scripts/run_benchmark_eval.py --model <model-or-path>
```

Use `--tasks` to subset (e.g. `--tasks piqa hellaswag`) and `--max-samples` for quicker sanity checks. The script scores each option by log-probability of the option text appended after the prompt. For PIQA, you can select the LM-Eval-Harness prompt with `--piqa-prompt-style lm_eval`.

Generative tasks (XSum summarization and WMT14 de->en translation) can be run with:

```
python scripts/run_benchmark_eval.py --model <model-or-path> --tasks xsum wmt14_de_en --max-samples 50
```

Optional generation controls include `--gen-max-new-tokens`, `--gen-do-sample`, `--gen-top-p`, and `--gen-temperature`.

python scripts/run_benchmark_eval.py   --model meta-llama/Llama-2-7b-hf   --tasks piqa  --mcq-prompt-style lm_eval --max-samples 100 --linear_layer LinearNF4Compute


## Benchmarks via lm-evaluation-harness

`run_benchmark_lm.py` uses the same model-loading and layer-replacement pipeline as `run_benchmark_eval.py`, but delegates all prompt formatting and scoring to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), producing numbers directly comparable to published results.

Install the harness once:

```bash
pip install lm-eval
```

Basic run (all supported tasks, 0-shot):

```bash
python scripts/run_benchmark_lm.py --model <model-or-path>
```

Run a subset of tasks with a specific layer variant:

```bash
python scripts/run_benchmark_lm.py \
    --model meta-llama/Llama-2-7b-hf \
    --tasks arc_easy arc_challenge hellaswag winogrande piqa \
    --linear_layer Linear4bit \
    --num-fewshot 0
```

Use the approximate BF16 matmul kernel (PRIM8 LUT):

```bash
python scripts/run_benchmark_lm.py \
    --model meta-llama/Llama-2-7b-hf \
    --tasks arc_easy arc_challenge hellaswag winogrande \
    --linear_layer LinearApproxBfloat16 \
    --prim8-lut aaR0
```

Save the full results JSON for later comparison:

```bash
python scripts/run_benchmark_lm.py \
    --model meta-llama/Llama-2-7b-hf \
    --linear_layer LinearNF4Compute \
    --output-path results_nf4compute.json
```

Supported flags:

- `--tasks` — space-separated subset from: `piqa siqa hellaswag winogrande arc_easy arc_challenge openbookqa commonsenseqa gsm8k humaneval xsum wmt14_de_en` (default: all).
- `--linear_layer` — same choices as `run_benchmark_eval.py`.
- `--prim8-lut` — PRIM8 LUT variant for `LinearApproxBfloat16` (default: `aaR0`).
- `--num-fewshot` — number of few-shot examples (default: 0).
- `--batch-size` — passed to lm-eval; `auto` lets it pick (default: `auto`).
- `--max-samples` — cap examples per task for quick sanity checks.
- `--output-path` — write the full lm-eval results dict as JSON.
