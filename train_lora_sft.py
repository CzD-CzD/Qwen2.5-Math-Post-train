from __future__ import annotations

import time
from typing import Any, Dict
import yaml
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import DatasetDict
from process_logger import log_event, LocalJSONLLogger


def load_lora_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_lora(
        model: Any,
        tokenizer: Any,
        train_ds: DatasetDict,
        val_ds: DatasetDict,
        yaml_path: str = "configs/lora_sft.yaml",
        resume_from_checkpoint: str | None = None,
):
    lora_cfg = load_lora_yaml(yaml_path)

    log_event(lora_cfg["run_name"], "config", lora_cfg)
    log_event(lora_cfg["run_name"], "dataset", {
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    })

    peft_config = LoraConfig(
        r=int(lora_cfg["lora_r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    args = SFTConfig(
        output_dir=lora_cfg["output_dir"],
        run_name=lora_cfg["run_name"],
        report_to=lora_cfg["report_to"],

        max_length=int(lora_cfg["max_length"]),
        num_train_epochs=int(lora_cfg["num_train_epochs"]),
        learning_rate=float(lora_cfg["learning_rate"]),
        per_device_train_batch_size=int(lora_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(lora_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(lora_cfg["gradient_accumulation_steps"]),

        logging_steps=int(lora_cfg["logging_steps"]),
        eval_strategy=lora_cfg["eval_strategy"],
        save_strategy=lora_cfg["save_strategy"],
        eval_steps=int(lora_cfg["eval_steps"]),
        save_total_limit=2,

        bf16=bool(lora_cfg.get("bf16", False)),
        fp16=bool(lora_cfg.get("fp16", False)),
        dataset_text_field=lora_cfg.get("dataset_text_field", "text"),

        remove_unused_columns=bool(lora_cfg["remove_unused_columns"]),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[LocalJSONLLogger(lora_cfg["run_name"])],
    )

    t0 = time.perf_counter()
    trainer.train()
    t1 = time.perf_counter()

    trainer.log({
        "train_time_sec": float(t1 - t0),
        "train_time_min": float(t1 - t0) / 60.0
    })
    log_event(lora_cfg["run_name"], "runtime", {
        "train_time_sec": float(t1 - t0)
    })

    trainer.save_model(lora_cfg["output_dir"])
    tokenizer.save_pretrained(lora_cfg["output_dir"])
