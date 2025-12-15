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
