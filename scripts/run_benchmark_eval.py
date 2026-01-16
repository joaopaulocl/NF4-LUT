#!/usr/bin/env python3
"""
Evaluate NF4-quantized LLAMA models on commonsense multiple-choice and
generative benchmarks.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
try:
    from bitsandbytes.nn import Linear4bit, Linear4bitFakeQuantAct, LinearNF4Compute
except ImportError:
    from bitsandbytes.nn import Linear4bit
    Linear4bitFakeQuantAct = None
    LinearNF4Compute = None

from bitsandbytes.functional import set_nf4_ewm_lut

@dataclass
class TaskSpec:
    dataset: str
    config: str | None
    split: str
    builder: Callable[[dict], tuple]
    trust_remote_code: bool = False
    task_type: str = "mcq"
    metric: str = "accuracy"


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


def xsum_builder(example: dict) -> tuple[str, str]:
    prompt = (
        "Summarize the following article.\n"
        f"Article: {example['document']}\n"
        "Summary:"
    )
    return prompt, example["summary"]


def wmt14_de_en_builder(example: dict) -> tuple[str, str]:
    source = example["translation"]["de"]
    target = example["translation"]["en"]
    prompt = (
        "Translate German to English.\n"
        f"German: {source}\n"
        "English:"
    )
    return prompt, target


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
    "xsum": TaskSpec(
        dataset="xsum",
        config=None,
        split="validation",
        builder=xsum_builder,
        task_type="gen",
        metric="rougeL",
    ),
    "wmt14_de_en": TaskSpec(
        dataset="wmt14",
        config="de-en",
        split="test",
        builder=wmt14_de_en_builder,
        task_type="gen",
        metric="bleu",
    ),
}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate NF4-quantized LLAMA models on benchmark tasks."
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
    parser.add_argument(
        "--lut-size",
        type=int,
        choices=[3, 4, 5, 6, 7, 8],
        default=8,
        help="NF4 EWM LUT size to use.",
    )
    parser.add_argument(
        "--linear_layer",
        choices=["Linear", "Linear4bit", "Linear4bitFakeQuantAct", "LinearNF4Compute"],
        default="Linear4bit",
        help="Type of linear layer to use for the model.",
    )
    parser.add_argument(
        "--gen-max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens for generative tasks.",
    )
    parser.add_argument(
        "--gen-do-sample",
        action="store_true",
        help="Enable sampling for generative tasks.",
    )
    parser.add_argument(
        "--gen-top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling value for generative tasks.",
    )
    parser.add_argument(
        "--gen-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generative tasks.",
    )
    return parser

def replace_linear4bit_with_fake(module):
    if Linear4bitFakeQuantAct is None:
        raise ImportError("Linear4bitFakeQuantAct is not available in this bitsandbytes build")
    for name, child in module.named_children():
        if isinstance(child, Linear4bit) and not isinstance(child, Linear4bitFakeQuantAct) and child.in_features in [4096, 2048, 1024, 512, 256, 128, 64]:
            new_layer = Linear4bitFakeQuantAct(
                input_features=child.in_features,
                output_features=child.out_features,
                bias=child.bias is not None,
                compute_dtype=child.compute_dtype,
                compress_statistics=getattr(child.weight, 'compress_statistics', True),
                quant_type=child.weight.quant_type,
                quant_storage=child.quant_storage,
                device=child.weight.device,
                activation_blocksize= child.in_features
            )
            new_layer.weight = child.weight  # Copy the Params4bit object
            new_layer.quant_state = child.quant_state
            if child.bias is not None:
                new_layer.bias = child.bias
            setattr(module, name, new_layer)
        else:
            replace_linear4bit_with_fake(child)


def replace_linear_with_fake(module):
    """Replace FP16 nn.Linear modules with Linear4bitFakeQuantAct.

    This starts from a non-quantized FP16 model and swaps selected Linear layers
    into bitsandbytes fake-activation-quant Linear4bit layers.
    """
    if Linear4bitFakeQuantAct is None:
        raise ImportError("Linear4bitFakeQuantAct is not available in this bitsandbytes build")

    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and child.in_features in [4096, 2048, 1024, 512, 256, 128, 64]:
            new_layer = Linear4bitFakeQuantAct(
                input_features=child.in_features,
                output_features=child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.float16,
                compress_statistics=False,
                quant_type="nf4",
                device=child.weight.device,
                activation_blocksize=64,
            )

            # Copy FP16 params over before quantization happens.
            with torch.no_grad():
                new_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data)

            # Trigger weight quantization inside Params4bit (see Params4bit.to()).
            new_layer = new_layer.to(child.weight.device)
            new_layer.eval()

            setattr(module, name, new_layer)
        else:
            replace_linear_with_fake(child)


def load_nf4_model(model_id: str, linear_layer: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if linear_layer == "Linear4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=quant_config
        )
        model.eval()
    elif linear_layer == "Linear4bitFakeQuantAct":
        # Start from a plain FP16 model, then replace nn.Linear layers.
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto"
        )
        model.eval()
        replace_linear_with_fake(model)
        model.eval()
    elif linear_layer == "LinearNF4Compute":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto"
        )
        model.eval()
        fp16_state = model.state_dict()

        def replace_linear_with_nf4(module, prefix=""):
            for name, child in module.named_children():
                child_prefix = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Linear) and child.in_features in [4096, 2048, 1024, 512, 256, 128, 64]:
                    new_layer = LinearNF4Compute(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=torch.float16,
                        compress_statistics=False,
                        #quant_storage=child.quant_storage,
                        device=child.weight.device,
                        blocksize=64,
                    )
                    # Try to load from the top-level fp16 state dict using the module prefix
                    weight_key = f"{child_prefix}.weight"
                    bias_key = f"{child_prefix}.bias"
                    if weight_key in fp16_state:
                        sd = {"weight": fp16_state[weight_key]}
                        if bias_key in fp16_state:
                            sd["bias"] = fp16_state[bias_key]
                    else:
                        # fallback to child's own parameters
                        print("Warning: could not find", weight_key, "in fp16 state dict")
                        child_sd = child.state_dict()
                        sd = {k: v for k, v in child_sd.items()}

                    # load into the NF4 compute layer (it will quantize internally)
                    new_layer.load_state_dict(sd, strict=False)
                    new_layer = new_layer.to(child.weight.device)
                    new_layer.eval()
                    setattr(module, name, new_layer)
                    
                else:
                    replace_linear_with_nf4(child, child_prefix)

        replace_linear_with_nf4(model)
        model.eval()

    else:   
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto"
        )

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


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower().strip())


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for token in a:
        curr = [0]
        for j, b_token in enumerate(b, start=1):
            if token == b_token:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(hypothesis: str, references: Sequence[str]) -> float:
    hyp_tokens = tokenize_text(hypothesis)
    if not hyp_tokens:
        return 0.0
    best = 0.0
    for ref in references:
        ref_tokens = tokenize_text(ref)
        if not ref_tokens:
            continue
        lcs = lcs_length(hyp_tokens, ref_tokens)
        precision = lcs / len(hyp_tokens) if hyp_tokens else 0.0
        recall = lcs / len(ref_tokens) if ref_tokens else 0.0
        if precision + recall == 0:
            score = 0.0
        else:
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def ngram_counts(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    if len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def update_bleu_stats(
    stats: dict[str, list[int] | int],
    hyp_tokens: list[str],
    references: Sequence[str],
) -> None:
    ref_tokens_list = [tokenize_text(ref) for ref in references]
    if not ref_tokens_list:
        return

    hyp_len = len(hyp_tokens)
    stats["hyp_len"] += hyp_len
    ref_len = min(ref_tokens_list, key=lambda r: (abs(len(r) - hyp_len), len(r)))
    stats["ref_len"] += len(ref_len)

    for n in range(1, 5):
        hyp_counts = ngram_counts(hyp_tokens, n)
        max_ref_counts: dict[tuple[str, ...], int] = {}
        for ref_tokens in ref_tokens_list:
            ref_counts = ngram_counts(ref_tokens, n)
            for ngram, count in ref_counts.items():
                if count > max_ref_counts.get(ngram, 0):
                    max_ref_counts[ngram] = count
        clipped = 0
        for ngram, count in hyp_counts.items():
            clipped += min(count, max_ref_counts.get(ngram, 0))
        stats["clipped"][n - 1] += clipped
        stats["total"][n - 1] += max(len(hyp_tokens) - n + 1, 0)


def compute_bleu(stats: dict[str, list[int] | int]) -> float:
    hyp_len = stats["hyp_len"]
    ref_len = stats["ref_len"]
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    log_precisions = 0.0
    for n in range(4):
        clipped = stats["clipped"][n]
        total = stats["total"][n]
        precision = (clipped + 1) / (total + 1)
        log_precisions += math.log(precision)
    bleu = bp * math.exp(log_precisions / 4)
    return bleu


def generate_completion(model, tokenizer, prompt: str, args: argparse.Namespace) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.gen_max_new_tokens,
            do_sample=args.gen_do_sample,
            top_p=args.gen_top_p,
            temperature=args.gen_temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = outputs[0][prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def evaluate_gen_task(
    name: str,
    spec: TaskSpec,
    model,
    tokenizer,
    max_samples: int | None,
    args: argparse.Namespace,
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

    iterator: Iterable[dict] = tqdm(dataset, desc=name, unit="ex")

    if spec.metric == "bleu":
        stats: dict[str, list[int] | int] = {
            "clipped": [0, 0, 0, 0],
            "total": [0, 0, 0, 0],
            "hyp_len": 0,
            "ref_len": 0,
        }
        for example in iterator:
            prompt, references = spec.builder(example)
            if isinstance(references, str):
                references = [references]
            hypothesis = generate_completion(model, tokenizer, prompt, args)
            update_bleu_stats(stats, tokenize_text(hypothesis), references)
        score = compute_bleu(stats)
        print(f"{name} BLEU: {score * 100:.2f}")
        return score

    total_score = 0.0
    for example in iterator:
        prompt, references = spec.builder(example)
        if isinstance(references, str):
            references = [references]
        hypothesis = generate_completion(model, tokenizer, prompt, args)
        total_score += rouge_l_f1(hypothesis, references)
    avg_score = total_score / total if total else 0.0
    print(f"{name} ROUGE-L: {avg_score * 100:.2f}")
    return avg_score


def gen_test(model, tokenizer):
    # Test with random input
    test_prompt = "Hello, how are you?"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                top_p=0.9,
                temperature=1.0,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"Test generation: {generated_text}")
    except Exception as e:
        print(f"Warning: Test generation failed with error: {e}")
        print("Proceeding with evaluation tasks...")
    return


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print("Setting NF4 EWM LUT size to", args.lut_size)
    set_nf4_ewm_lut(args.lut_size) 
    model, tokenizer = load_nf4_model(args.model, args.linear_layer)
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"{name} contains NaN or inf")

    #gen_test(model, tokenizer)
    
    if "all" in args.tasks:
        task_names = list(TASK_SPECS.keys())
    else:
        task_names = args.tasks

    results = {}
    for task_name in task_names:
        spec = TASK_SPECS[task_name]
        if spec.task_type == "gen":
            score = evaluate_gen_task(
                task_name,
                spec,
                model,
                tokenizer,
                args.max_samples,
                args,
            )
        else:
            score = evaluate_task(task_name, spec, model, tokenizer, args.max_samples)
        results[task_name] = score

    print("\nSummary:")
    for task_name in task_names:
        spec = TASK_SPECS[task_name]
        score = results[task_name]
        if spec.metric == "accuracy":
            print(f"- {task_name}: {score * 100:.2f}%")
        elif spec.metric == "bleu":
            print(f"- {task_name} (BLEU): {score * 100:.2f}")
        else:
            print(f"- {task_name} (ROUGE-L): {score * 100:.2f}")


if __name__ == "__main__":
    main()
