import argparse
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope.hub.snapshot_download import snapshot_download
from peft import PeftModel

from data_gsm8k import get_gsm8k_ds
from data_math import get_math_ds
from train_full_sft import train_sft
from train_lora_sft import train_lora
from train_prompt_sft import train_prompt_tuning

from infer_basic import infer_basic
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
    p.add_argument("--mode", choices=["sft", "lora", "prompt"], required=True)
    p.add_argument("--model", choices=MODELS.keys(), required=True)
    p.add_argument("--model-path", default=None, help="Local model path (overrides --model)")
    p.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    return p.parse_args()


def main():
    args = parse_args()

    ds = DATASETS[args.dataset]()
    train_ds, val_ds, test_ds = ds["train"], ds["val"], ds["test"]
    # test_ds = ds["test"].select(range(1))

    if args.task == "train":

        model_dir = snapshot_download(model_id=MODELS[args.model])
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )

        if args.mode == "sft":
            train_sft(model, tokenizer, train_ds, val_ds)
        elif args.mode == "lora":
            train_lora(model, tokenizer, train_ds, val_ds)
        elif args.mode == "prompt":
            train_prompt_tuning(model, tokenizer, train_ds, val_ds)
        else:
            raise ValueError(f"train task does not support mode={args.mode}")
        
    elif args.task == "infer":
        base_model_dir = snapshot_download(model_id=MODELS[args.model])
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
            attn_implementation="eager" if args.mode in ["prefix", "prompt"] else None,
        )
    
        if args.model_path:
            if args.mode in ["lora", "prompt"]:
                model = PeftModel.from_pretrained(base_model, args.model_path)
                tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
            elif args.mode == "sft":
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    trust_remote_code=True,
                    dtype="auto",
                    device_map="auto",
                )
                tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            else:
                raise ValueError(f"infer task does not support mode={args.mode}")
        else:
            model = base_model
            tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    
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
