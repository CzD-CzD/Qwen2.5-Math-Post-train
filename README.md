# PEFT: Qwen2.5-0.5B GSM8K Finetune (SFT / LoRA / Prompt)

本项目用于对比 Qwen2.5-0.5B 在 GSM8K 上的不同 PEFT 微调方式（目前不包含 prefix 方式的对比）。

## 环境
- Python 3.10
- PyTorch 2.5.1+cu121
- CUDA 12.1

## 目录说明
- `run_experiments.py`: 统一入口，按模式启动训练/推理
- `run_all.sh`: 批量运行脚本
- `train_lora_sft.py` / `train_prompt_sft.py` / `train_prefix_sft.py`: 不同训练方式
- `eval_gsm8k.py`: 评测 GSM8K 结果
- `data_gsm8k.py`: 数据处理
- `dataset/`: 数据目录（请放置 GSM8K 数据）
- `infer_outputs/`: 推理输出（jsonl）

## 快速开始
安装依赖：
```bash
pip install -r requirements.txt
```

运行实验（示例）：
```bash
python run_experiments.py --mode sft --model 0.5B --dataset gsm8k
python run_experiments.py --mode lora --model 0.5B --dataset gsm8k
python run_experiments.py --mode prompt --model 0.5B --dataset gsm8k
```

评测：
```bash
python eval_gsm8k.py infer_outputs/base_infer.jsonl
python eval_gsm8k.py infer_outputs/sft_infer.jsonl
python eval_gsm8k.py infer_outputs/lora_infer.jsonl
python eval_gsm8k.py infer_outputs/prompt_infer.jsonl
```

## 当前结果（GSM8K）
- base: total=1318 correct=398 acc=0.3020
- sft: total=1319 correct=478 acc=0.3624
- lora: total=1319 correct=468 acc=0.3548
- prompt: total=1318 correct=405 acc=0.3073

## 说明
- 当前暂未做 prefix 对比。
