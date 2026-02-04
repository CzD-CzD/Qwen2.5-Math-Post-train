from __future__ import annotations

import time
from typing import Dict, Any
import yaml

from trl import SFTTrainer, SFTConfig
from datasets import DatasetDict
from peft import PromptTuningConfig, get_peft_model

from process_logger import log_event, LocalJSONLLogger


def load_prompt_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_prompt_tuning(
        model: Any,
        tokenizer: Any,
        train_ds: DatasetDict,
        val_ds: DatasetDict,
        yaml_path: str = "configs/prompt_tuning.yaml",
):
    prompt_cfg = load_prompt_yaml(yaml_path)

    log_event(prompt_cfg["run_name"], "config", prompt_cfg)
    log_event(prompt_cfg["run_name"], "dataset", {
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    })

    peft_config = PromptTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=int(prompt_cfg["num_virtual_tokens"]),
    )
    model = get_peft_model(model, peft_config)
    model.config.sue_cache = False

    args = SFTConfig(
        output_dir=prompt_cfg["output_dir"],
        run_name=prompt_cfg["run_name"],
        report_to=prompt_cfg["report_to"],

        max_length=int(prompt_cfg["max_length"]),
        num_train_epochs=int(prompt_cfg["num_train_epochs"]),
        learning_rate=float(prompt_cfg["learning_rate"]),
        per_device_train_batch_size=int(prompt_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(prompt_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(prompt_cfg["gradient_accumulation_steps"]),

        logging_steps=int(prompt_cfg["logging_steps"]),
        eval_strategy=prompt_cfg["eval_strategy"],
        save_strategy=prompt_cfg["save_strategy"],
        eval_steps=int(prompt_cfg["eval_steps"]),
        save_total_limit=2,

        bf16=bool(prompt_cfg.get("bf16", False)),
        fp16=bool(prompt_cfg.get("fp16", False)),
        dataset_text_field=prompt_cfg.get("dataset_text_field", "text"),

        remove_unused_columns=bool(prompt_cfg["remove_unused_columns"]),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[LocalJSONLLogger(prompt_cfg["run_name"])],
    )

    t0 = time.perf_counter()
    trainer.train()
    t1 = time.perf_counter()

    trainer.log({
        "train_time_sec": float(t1 - t0),
        "train_time_min": float(t1 - t0) / 60.0
    })
    log_event(prompt_cfg["run_name"], "runtime", {"train_time_sec": float(t1 - t0)})

    trainer.save_model(prompt_cfg["output_dir"])
    tokenizer.save_pretrained(prompt_cfg["output_dir"])
