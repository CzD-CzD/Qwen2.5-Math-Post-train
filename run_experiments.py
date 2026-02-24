import argparse
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope.hub.snapshot_download import snapshot_download
from peft import PeftModel

from data.data_gsm8k import get_gsm8k_ds
from data.data_math import get_math_ds
from data.data_deepscaler import get_deepscaler_ds
from sft.train_full_sft import train_sft
from sft.train_lora_sft import train_lora
from sft.train_prompt_sft import train_prompt_tuning
from rl.train_grpo import train_grpo

from inference.infer_basic import infer_basic
#from infer_bon import infer_bon

DATASETS = {
    "gsm8k": get_gsm8k_ds,
    "math": get_math_ds,
}

MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B",
    "1.5B": "Qwen/Qwen2.5-Math-1.5B",
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["train", "infer"], required=True)
    p.add_argument("--infer-mode", choices=["basic", "bon", "vote"], help="infer mode")
    p.add_argument("--mode", choices=["sft", "lora", "prompt", "grpo"], required=True)
    p.add_argument("--model", choices=MODELS.keys(), required=True)
    p.add_argument("--model-path", default=None, help="Local model path (overrides --model)")
    p.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    #p.add_argument("--use-diet", action="store_true")
    return p.parse_args()

def load_base(model_key: str, mode: str):
    base_model_dir = snapshot_download(model_id=MODELS[model_key])
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
        attn_implementation="eager" if mode == "prompt" else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    return model, tokenizer, base_model_dir

def load_adapter(base_model, base_model_dir: str, mode: str, model_path: str | None):
    if not model_path:
        return base_model, AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    if mode in ["lora", "prompt", "grpo"]:
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=(mode in ["lora", "grpo"]))
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
        return model, tokenizer
    if mode == "sft":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer
    raise ValueError(f"unsupported mode for model_path: {mode}")

def main():
    args = parse_args()

    ds = DATASETS[args.dataset]()
    train_ds, val_ds, test_ds = ds["train"], ds["val"], ds["test"]
    # test_ds = ds["test"].select(range(1))

    if args.task == "train":

        base_model, tokenizer, base_model_dir = load_base(args.model, args.mode)

        if args.mode == "sft":
            train_sft(base_model, tokenizer, train_ds, val_ds)
        elif args.mode == "lora":
            model, tokenizer = load_adapter(base_model, base_model_dir, "lora", args.model_path)
            train_lora(model, tokenizer, train_ds, val_ds)
        elif args.mode == "prompt":
            model, tokenizer = load_adapter(base_model, base_model_dir, "prompt", args.model_path)
            train_prompt_tuning(model, tokenizer, train_ds, val_ds)

        # only support peft + grpo, please change the code below when use sft
        elif args.mode == "grpo":
            if not args.model_path:
                raise ValueError("GRPO requires a local model path (--model-path)")
            model, tokenizer = load_adapter(base_model, base_model_dir, "grpo", args.model_path)
            train_grpo(model, tokenizer, train_ds, val_ds)
        else:
            raise ValueError(f"train task does not support mode={args.mode}")
        
    elif args.task == "infer":
        base_model, tokenizer, base_model_dir = load_base(args.model, args.mode)
        model, tokenizer = load_adapter(base_model, base_model_dir, args.mode, args.model_path)
    
        if args.infer_mode == "basic":
            infer_basic(model, tokenizer, test_ds)
        #elif args.infer_mode == "bon":
        #    infer_bon(model, tokenizer, test_ds)
        else:
            raise ValueError(f"infer task does not support infer_mode={args.infer_mode}")

    else:
        raise ValueError(f"Unknown task={args.task}")


if __name__ == "__main__":
    main()
