# Qwen2.5 Math Finetune (SFT / LoRA / Prompt)

本项目用于对比 Qwen2.5-0.5B/Qwen2.5-Math-1.5B 在 GSM8K 上的不同 PEFT 微调方式。
项目使用了 SwanLab 进行实验记录与可视化。

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
python run_experiments.py --task train --mode sft --model 1.5B --dataset gsm8k
python run_experiments.py --task train --mode lora --model 1.5B --dataset gsm8k
python run_experiments.py --task train --mode prompt --model 1.5B --dataset gsm8k
python run_experiments.py --task train --mode grpo --model 1.5B --dataset gsm8k --model-path ./out/lora_math
python run_experiments.py --task infer --mode lora --infer-mode basic --model 1.5B --dataset gsm8k --model-path ./out/grpo_math
```

评测：
```bash
python eval_gsm8k.py infer_outputs/base_infer.jsonl
python eval_gsm8k.py infer_outputs/sft_infer.jsonl
python eval_gsm8k.py infer_outputs/lora_infer.jsonl
python eval_gsm8k.py infer_outputs/prompt_infer.jsonl
```

## Results（GSM8K, Qwen2.5-0.5B）
- base: total=1318 correct=398 acc=0.3020
- sft: total=1319 correct=478 acc=0.3624
- lora: total=1319 correct=468 acc=0.3548
- prompt: total=1319 correct=405 acc=0.3071


## Results (GSM8K, Qwen2.5-Math-1.5B)
- base: total=1319 correct=835 acc=0.6331 (format_acc=0.7043)
- sft: total=1319 correct=939 acc=0.7119 (format_acc=0.9901)
- lora(r=16, q+v, alpha=32): total=1319 correct=887 acc=0.6725 (format_acc=0.9939)
- lora(r=16, q+v+k, alpha=32): total=1319 correct=912 acc=0.6914 (format_acc=0.9901)
- lora(r=16, q+v+k+o, alpha=32): total=1319 correct=905 acc=0.6861 (format_acc=0.9886)
- lora(r=16, q+v+k, alpha=64): total=1319 correct=899 acc=0.6816 (format_acc=0.9871)
- lora(r=8, q+v+k, alpha=32): total=1319 correct=892 acc=0.6763 (format_acc=0.9886)
- grpo(lora best, kl_coef=0.0): total=1319 correct=996 acc=0.7551 (format_acc=0.9962)


