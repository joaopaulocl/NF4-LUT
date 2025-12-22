#!/usr/bin/env python3
"""
Estimate perplexity on WikiText-2 using an NF4-quantized LLAMA model.
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute perplexity on WikiText-2 using an NF4-quantized model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name or path compatible with AutoModelForCausalLM.",
    )
    parser.add_argument(
        "--dataset-name",
        default="wikitext",
        help="HF dataset name (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset-config",
        default="wikitext-2-raw-v1",
        help="HF dataset config (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length passed to the model (default: %(default)s).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1024,
        help="Sliding window stride between chunks (default: %(default)s).",
    )
    parser.add_argument(
        "--max-texts",
        type=int,
        default=None,
        help="Optional cap on number of text samples to evaluate.",
    )
    return parser


def load_nf4_model(model_path: str):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", 
        quantization_config=quant_config
    )
    return model, tokenizer


def iter_texts(args: argparse.Namespace) -> Iterable[str]:
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    texts = ds["text"]
    if args.max_texts is not None:
        texts = texts[: args.max_texts]
    for text in texts:
        if text and not text.isspace():
            yield text


def accumulate_nll(
    model,
    tokenizer,
    texts: Iterable[str],
    max_length: int,
    stride: int,
) -> tuple[float, int]:
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        tokenized = tokenizer(text, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(model.device)
        seq_len = input_ids.size(1)

        for i in range(0, seq_len, stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, seq_len)
            target_len = end_loc - i
            if target_len <= 0:
                continue

            input_chunk = input_ids[:, begin_loc:end_loc]
            target_ids = input_chunk.clone()
            target_ids[:, :-target_len] = -100

            with torch.inference_mode():
                outputs = model(input_chunk, labels=target_ids)

            total_nll += outputs.loss.item() * target_len
            total_tokens += target_len

    return total_nll, total_tokens


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    model, tokenizer = load_nf4_model(args.model)

    texts = iter_texts(args)
    total_nll, total_tokens = accumulate_nll(
        model, tokenizer, texts, args.max_length, args.stride
    )

    if total_tokens == 0:
        raise RuntimeError("No tokens were evaluated; check dataset settings.")

    ppl = math.exp(total_nll / total_tokens)
    print(
        f"Perplexity on {args.dataset_name}/{args.dataset_config}:{args.split} "
        f"(max_len={args.max_length}, stride={args.stride}) = {ppl:.2f}"
    )


if __name__ == "__main__":
    main()
