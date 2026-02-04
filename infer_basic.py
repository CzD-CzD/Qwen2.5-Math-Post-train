from __future__ import annotations

import json
from typing import Iterable, Optional, Dict, Any, List
import torch
import yaml
import os


def load_infer_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _chunked(it: List[dict], n: int):
    for i in range(0, len(it), n):
        yield it[i:i + n]

def wrap_chat(tokenizer, p):
    messages = [
        {"role": "system", "content": "You are a math assistant who solves problems step by step."},
        {"role": "user", "content": p},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def infer_basic(
    model,
    tokenizer,
    dataset: Iterable[dict],
    yaml_path: str = "configs/infer_basic.yaml",
    seed: int = 42,
):
    cfg = load_infer_yaml(yaml_path)

    output_dir = cfg.get("output_dir", "infer_outputs")
    output_name = os.environ.get("INFER_OUTPUT_NAME") or cfg.get("output_name", "infer.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    batch_size = int(cfg.get("batch_size", 1))

    max_new_tokens = int(cfg.get("max_new_tokens", 256))
    # temperature = float(cfg.get("temperature", 0.7))
    # top_p = float(cfg.get("top_p", 0.9))
    do_sample = bool(cfg.get("do_sample", False))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    data_list = list(dataset)
    with open(output_path, "w", encoding="utf-8") as f:
        for batch in _chunked(data_list, batch_size):
            # prompts = [wrap_chat(tokenizer, ex["prompt"]) for ex in batch]
            prompts = [ex["prompt"] for ex in batch]



            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}


            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    #no_repeat_ngram_size=3,
                    #repetition_penalty=1.1,
                    #temperature=temperature,
                    #top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id, 
                    eos_token_id=tokenizer.eos_token_id,
                )


            input_lens = inputs["attention_mask"].sum(dim=1)
            for i, ex in enumerate(batch):
                start = int(input_lens[i])
                gen_ids = outputs[i, start:]
                pred = tokenizer.decode(gen_ids, skip_special_tokens=True)

                record = {
                    "question": ex.get("question"),
                    #"prompt": ex.get("prompt"),
                    "prediction": pred,
                    "gold": ex.get("gold"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[infer_basic] wrote outputs to {output_path}")