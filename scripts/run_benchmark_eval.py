#!/usr/bin/env python3
"""
Evaluate NF4-quantized LLAMA models on commonsense multiple-choice benchmarks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class TaskSpec:
    dataset: str
    config: str | None
    split: str
    builder: Callable[[dict], tuple[str, list[str], int]]
    trust_remote_code: bool = False


LETTERS = ["A", "B", "C", "D", "E", "F"]


def make_prompt(instruction: str, body: str, options: Sequence[tuple[str, str]]) -> str:
    """Formats a generic multiple-choice question prompt."""
    lines: list[str] = []
    if instruction:
        lines.append(instruction.strip())
    lines.append(body.strip())
    lines.append("Options:")
    for label, text in options:
        lines.append(f"{label}. {text}")
    lines.append("Answer:")
    return "\n".join(lines)


def piqa_builder(example: dict) -> tuple[str, list[str], int]:
    options = [(LETTERS[i], example[f"sol{i + 1}"]) for i in range(2)]
    prompt = make_prompt(
        "Choose the option that best completes the goal.",
        f"Goal: {example['goal']}",
        options,
    )
    return prompt, [label for label, _ in options], int(example["label"])


def siqa_builder(example: dict) -> tuple[str, list[str], int]:
    options = [
        (LETTERS[0], example["answerA"]),
        (LETTERS[1], example["answerB"]),
        (LETTERS[2], example["answerC"]),
    ]
    prompt = make_prompt(
        "Given the situation, pick the most sensible answer.",
        f"Context: {example['context']}\nQuestion: {example['question']}",
        options,
    )
    return prompt, [label for label, _ in options], int(example["label"]) - 1


def hellaswag_builder(example: dict) -> tuple[str, list[str], int]:
    options = [(LETTERS[i], ending) for i, ending in enumerate(example["endings"])]
    prompt = make_prompt(
        "Choose the most plausible continuation.",
        f"Premise: {example['ctx']}",
        options,
    )
    return prompt, [label for label, _ in options], int(example["label"])


def winogrande_builder(example: dict) -> tuple[str, list[str], int]:
    base_sentence = example["sentence"].replace("_", "_____")
    options = [
        (LETTERS[0], example["option1"]),
        (LETTERS[1], example["option2"]),
    ]
    prompt = make_prompt(
        "Fill in the blank in the sentence with the most sensible option.",
        base_sentence,
        options,
    )
    return prompt, [label for label, _ in options], int(example["answer"]) - 1


def arc_builder(example: dict) -> tuple[str, list[str], int]:
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    options = [(label, text) for label, text in zip(labels, texts)]
    prompt = make_prompt(
        "Pick the correct answer to the science exam question.",
        example["question"],
        options,
    )
    correct_idx = labels.index(example["answerKey"])
    return prompt, [label for label, _ in options], correct_idx


def openbookqa_builder(example: dict) -> tuple[str, list[str], int]:
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    options = [(label, text) for label, text in zip(labels, texts)]
    prompt = make_prompt(
        "Answer the elementary science question.",
        example["question_stem"],
        options,
    )
    answer_idx = labels.index(example["answerKey"])
    return prompt, [label for label, _ in options], answer_idx


def commonsenseqa_builder(example: dict) -> tuple[str, list[str], int]:
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    options = [(label, text) for label, text in zip(labels, texts)]
    prompt = make_prompt(
        "Answer the commonsense question.",
        example["question"],
        options,
    )
    answer_idx = labels.index(example["answerKey"])
    return prompt, [label for label, _ in options], answer_idx


TASK_SPECS: dict[str, TaskSpec] = {
    "piqa": TaskSpec(
        dataset="piqa",
        config=None,
        split="validation",
        builder=piqa_builder,
        trust_remote_code=True,
    ),
    "siqa": TaskSpec(
        dataset="social_i_qa",
        config=None,
        split="validation",
        builder=siqa_builder,
        trust_remote_code=True,
    ),
    "hellaswag": TaskSpec(
        dataset="hellaswag",
        config=None,
        split="validation",
        builder=hellaswag_builder,
    ),
    "winogrande": TaskSpec(
        dataset="winogrande",
        config="winogrande_xl",
        split="validation",
        builder=winogrande_builder,
    ),
    "arc_easy": TaskSpec(
        dataset="ai2_arc",
        config="ARC-Easy",
        split="validation",
        builder=arc_builder,
    ),
    "arc_challenge": TaskSpec(
        dataset="ai2_arc",
        config="ARC-Challenge",
        split="validation",
        builder=arc_builder,
    ),
    "openbookqa": TaskSpec(
        dataset="openbookqa",
        config="main",
        split="validation",
        builder=openbookqa_builder,
    ),
    "commonsenseqa": TaskSpec(
        dataset="commonsense_qa",
        config=None,
        split="validation",
        builder=commonsenseqa_builder,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate NF4-quantized LLAMA models on benchmark QA tasks."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name or path compatible with AutoModelForCausalLM.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["all", *TASK_SPECS.keys()],
        default=["all"],
        help="Tasks to run (default: all).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of examples per task.",
    )
    return parser


def load_nf4_model(model_id: str):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", quantization_config=quant_config
    )
    model.eval()
    return model, tokenizer


def score_option(
    model,
    tokenizer,
    prompt_ids: torch.LongTensor,
    completion: str,
) -> float:
    completion_ids = tokenizer(
        completion, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(model.device)
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    return -outputs.loss.item() * completion_ids.shape[1]


def evaluate_task(
    name: str,
    spec: TaskSpec,
    model,
    tokenizer,
    max_samples: int | None,
) -> float:
    dataset = load_dataset(
        spec.dataset,
        spec.config,
        split=spec.split,
        trust_remote_code=spec.trust_remote_code,
    )
    total = len(dataset)
    if max_samples is not None:
        total = min(total, max_samples)
        dataset = dataset.select(range(total))

    correct = 0
    iterator: Iterable[dict] = tqdm(dataset, desc=name, unit="ex")

    for example in iterator:
        prompt, labels, answer_idx = spec.builder(example)
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_ids = prompt_inputs["input_ids"]
        completions = [f" {label}" for label in labels]
        scores = [
            score_option(model, tokenizer, prompt_ids, completion)
            for completion in completions
        ]
        pred_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        correct += int(pred_idx == answer_idx)

    accuracy = correct / total if total else 0.0
    print(f"{name}: {accuracy * 100:.2f}% ({correct}/{total})")
    return accuracy


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    model, tokenizer = load_nf4_model(args.model)

    if "all" in args.tasks:
        task_names = list(TASK_SPECS.keys())
    else:
        task_names = args.tasks

    results = {}
    for task_name in task_names:
        spec = TASK_SPECS[task_name]
        acc = evaluate_task(task_name, spec, model, tokenizer, args.max_samples)
        results[task_name] = acc

    print("\nSummary:")
    for task_name in task_names:
        acc = results[task_name]
        print(f"- {task_name}: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
