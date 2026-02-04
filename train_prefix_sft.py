from __future__ import annotations

import time
from typing import Any, Dict
import yaml
from trl import SFTTrainer, SFTConfig
from peft import PrefixTuningConfig, get_peft_model
from datasets import DatasetDict
from process_logger import log_event, LocalJSONLLogger


def load_prefix_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_prefix(
        model: Any,
        tokenizer: Any,
        train_ds: DatasetDict,
        val_ds: DatasetDict,
        yaml_path: str = "configs/prefix_sft.yaml",
):
    prefix_cfg = load_prefix_yaml(yaml_path)

    log_event(prefix_cfg["run_name"], "config", prefix_cfg)
    log_event(prefix_cfg["run_name"], "dataset", {
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    })

    peft_config = PrefixTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=int(prefix_cfg["num_virtual_tokens"]),
        prefix_projection=bool(prefix_cfg.get("prefix_projection", False)),
    )

    model = get_peft_model(model, peft_config)
    model.config.use_cache = False

    args = SFTConfig(
        output_dir=prefix_cfg["output_dir"],
        run_name=prefix_cfg["run_name"],
        report_to=prefix_cfg["report_to"],

        max_length=int(prefix_cfg["max_length"]),
        num_train_epochs=int(prefix_cfg["num_train_epochs"]),
        learning_rate=float(prefix_cfg["learning_rate"]),
        per_device_train_batch_size=int(prefix_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(prefix_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(prefix_cfg["gradient_accumulation_steps"]),

        logging_steps=int(prefix_cfg["logging_steps"]),
        eval_strategy=prefix_cfg["eval_strategy"],
        save_strategy=prefix_cfg["save_strategy"],
        eval_steps=int(prefix_cfg["eval_steps"]),
        save_total_limit=2,

        bf16=bool(prefix_cfg.get("bf16", False)),
        fp16=bool(prefix_cfg.get("fp16", False)),
        dataset_text_field=prefix_cfg.get("dataset_text_field", "text"),

        remove_unused_columns=bool(prefix_cfg["remove_unused_columns"]),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[LocalJSONLLogger(prefix_cfg["run_name"])],
    )


    t0 = time.perf_counter()
    trainer.train()
    t1 = time.perf_counter()

    trainer.log({
        "train_time_sec": float(t1 - t0),
        "train_time_min": float(t1 - t0) / 60.0
    })
    log_event(prefix_cfg["run_name"], "runtime", {
        "train_time_sec": float(t1 - t0)
    })

    trainer.save_model(prefix_cfg["output_dir"])
    tokenizer.save_pretrained(prefix_cfg["output_dir"])
