import json, os, time
from typing import Any, Dict
from transformers import TrainerCallback

class LocalJSONLLogger(TrainerCallback):
    def __init__(self, run_name: str):
        self.run_name = run_name

    def on_train_begin(self, args, state, control, **kwargs):
        log_event(self.run_name, "train_begin", {
            "global_step": state.global_step,
            "max_steps": state.max_steps,
        })

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log_event(self.run_name, "train_log", logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            log_event(self.run_name, "eval_log", metrics)

    def on_train_end(self, args, state, control, **kwargs):
        log_event(self.run_name, "train_end", {
            "global_step": state.global_step
        })



def log_event(run_name: str, event: str, payload: Dict[str, Any] | None = None):
    os.makedirs("process-log", exist_ok=True)

    record = {
        "ts": time.time(),
        "event": event,
        "payload": payload or {},
    }

    path = os.path.join("process-log", f"{run_name}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str
            ) + "\n"
        )
        f.flush()
