from __future__ import annotations

import time
import os
from typing import Optional, Dict, Any
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import DatasetDict
from dotenv import load_dotenv
import swanlab
from process_logger import log_event, LocalJSONLLogger

load_dotenv()

def load_sft_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_sft(
        model: Any,
        tokenizer: Any,
        train_ds: DatasetDict,
        val_ds: DatasetDict,
        yaml_path: str = "configs/full_sft.yaml",
):
    sft_cfg = load_sft_yaml(yaml_path)

    log_event(sft_cfg["run_name"], "config", sft_cfg)
    log_event(sft_cfg["run_name"], "dataset", {
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    })

    response_template = "Solution:\n" 

    args = SFTConfig(
        output_dir=sft_cfg["output_dir"],
        run_name=sft_cfg["run_name"],
        report_to=sft_cfg["report_to"],

        max_length=int(sft_cfg["max_length"]),
        num_train_epochs=int(sft_cfg["num_train_epochs"]),
        learning_rate=float(sft_cfg["learning_rate"]),
        per_device_train_batch_size=int(sft_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(sft_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(sft_cfg["gradient_accumulation_steps"]),

        logging_steps=int(sft_cfg["logging_steps"]),
        eval_strategy=sft_cfg["eval_strategy"],
        save_strategy=sft_cfg["save_strategy"],
        eval_steps=int(sft_cfg["eval_steps"]),
        save_total_limit=2,


        bf16=bool(sft_cfg.get("bf16", False)),
        fp16=bool(sft_cfg.get("fp16", False)),
        dataset_text_field=sft_cfg.get("dataset_text_field", "text"),

        remove_unused_columns=bool(sft_cfg["remove_unused_columns"]),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[LocalJSONLLogger(sft_cfg["run_name"])],
    )

    t0 = time.perf_counter()
    trainer.train()
    t1 = time.perf_counter()

    trainer.log({"train_time_sec": float(t1 - t0), "train_time_min": float(t1 - t0) / 60.0})
    log_event(sft_cfg["run_name"], "runtime", {"train_time_sec": float(t1 - t0)})

    trainer.save_model(sft_cfg["output_dir"])
    tokenizer.save_pretrained(sft_cfg["output_dir"])
