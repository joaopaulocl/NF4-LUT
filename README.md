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