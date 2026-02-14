from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

import yaml
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from utils.reward_math import extract_answer_strict, extract_boxed_answer, grade


def load_grpo_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict) and "content" in x:
        return str(x["content"])
    if isinstance(x, list) and x and isinstance(x[-1], dict) and "content" in x[-1]:
        return str(x[-1]["content"])
    return str(x)


def _clean_answer(s: str) -> str:
    s = s.strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    s = re.sub(r"[.,;:!?]+$", "", s).strip()
    return s


def _find_references(kwargs: Dict[str, Any], n: int, m: int) -> List[Optional[str]]:
    value = kwargs.get("gold")
    if value is None:
        return [None] * n
    if isinstance(value, list):
        if len(value) == n:
            return [None if v is None else str(v) for v in value]
        if m > 0 and len(value) == m and n % m == 0:
            k = n // m
            out: List[Optional[str]] = []
            for v in value:
                out.extend([None if v is None else str(v)] * k)
            return out
        return [None] * n
    return [str(value)] * n


def build_reward_fn() -> Callable[..., List[float]]:
    def reward_fn(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
        refs = _find_references(kwargs, len(completions), len(prompts))
        rewards: List[float] = []

        for i, completion in enumerate(completions):
            text = _to_text(completion)
            ans = extract_answer_strict(text)
            if ans is None:
                rewards.append(0.0)
                continue
            ans = _clean_answer(ans)
            if "\\boxed" in ans:
                boxed = extract_boxed_answer(ans)
                if boxed is not None:
                    ans = boxed
            gold = refs[i]
            rewards.append(1.0 if grade(ans, gold, fast=True) else 0.0)
        return rewards

    return reward_fn


def train_grpo(
    model: Any,
    tokenizer: Any,
    train_ds: Dataset,
    val_ds: Optional[Dataset] = None,
    yaml_path: str = "configs/grpo.yaml",
) -> None:
    cfg = load_grpo_yaml(yaml_path)
    grpo_cfg = cfg.get("grpo", cfg)


    args = GRPOConfig(
        output_dir=grpo_cfg["output_dir"],
        run_name=grpo_cfg.get("run_name"),
        report_to=grpo_cfg.get("report_to"),
        
        learning_rate=float(grpo_cfg["learning_rate"]),

        per_device_train_batch_size=int(grpo_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(grpo_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(grpo_cfg["gradient_accumulation_steps"]),
        max_steps=int(grpo_cfg["max_steps"]),
        num_generations_eval=int(grpo_cfg["num_generations_eval"]),

        logging_steps=int(grpo_cfg["logging_steps"]),
        eval_strategy=grpo_cfg["eval_strategy"],
        eval_steps=int(grpo_cfg["eval_steps"]),
        save_strategy=grpo_cfg["save_strategy"],
        save_steps=grpo_cfg["save_steps"],

        max_completion_length=int(grpo_cfg["max_completion_length"]),
        num_generations=int(grpo_cfg["num_generations"]),
        generation_batch_size=int(grpo_cfg["generation_batch_size"]),
        beta=float(grpo_cfg["beta"]),

        bf16=bool(grpo_cfg.get("bf16", False)),
        fp16=bool(grpo_cfg.get("fp16", False)),
        remove_unused_columns=bool(grpo_cfg["remove_unused_columns"]),
    )
    reward_fn = build_reward_fn()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
