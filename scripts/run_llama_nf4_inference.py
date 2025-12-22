#!/usr/bin/env python3
"""
Simple LLAMA inference script that loads a checkpoint with bitsandbytes NF4 quantization.
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LLAMA inference with bitsandbytes NF4 quantization."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name or path compatible with AutoModelForCausalLM.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain NF4 quantization in one paragraph.",
        help="Prompt text to feed to the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling value (ignored if do_sample is False).",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling. If disabled, greedy decoding is used.",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Wrap the prompt with the tokenizer chat template (if available).",
    )
    return parser


def maybe_format_prompt(
    tokenizer, prompt: str, use_chat_template: bool
) -> tuple[str, Optional[list[dict]]]:
    """
    Returns a string prompt, and the chat messages if one was used.
    """
    if not use_chat_template:
        return prompt, None

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "Tokenizer does not support chat templates. Remove --use-chat-template."
        )

    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return rendered, messages


def load_model(model_path: str):
    """
    Loads a LLAMA model with NF4 quantization via bitsandbytes.
    """
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto"
        #model_path, quantization_config=quant_config, device_map="auto"
    )
    return model, tokenizer


def run_inference(args: argparse.Namespace) -> None:
    model, tokenizer = load_model(args.model)
    prompt_text, _ = maybe_format_prompt(tokenizer, args.prompt, args.use_chat_template)
    inputs = tokenizer(
        prompt_text, return_tensors="pt", padding=True
    ).to(model.device)

    generate_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.inference_mode():
        output = model.generate(**inputs, **generate_kwargs)

    response_tokens = output[0, inputs["input_ids"].shape[-1] :]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    print("Prompt:\n", args.prompt)
    print("\nResponse:\n", response_text.strip())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
