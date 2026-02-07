#!/usr/bin/env bash
set -euo pipefail

# 基础参数
MODEL=1.5B
DATASET=gsm8k   # 你需要的话可以改成 math
PY=python

# 日志目录
LOG_DIR=logs
mkdir -p "${LOG_DIR}"

# 0) Base
#INFER_OUTPUT_NAME="base_infer_bon.jsonl" ${PY} run_experiments.py --task infer --infer-mode bon --mode sft --model ${MODEL} --dataset ${DATASET} \
#  > "${LOG_DIR}/base_infer_bon.log" 2>&1

#INFER_OUTPUT_NAME="base_infer_basic.jsonl" ${PY} run_experiments.py --task infer --infer-mode basic --mode sft --model ${MODEL} --dataset ${DATASET} \
#  > "${LOG_DIR}/base_infer_basic.log" 2>&1
  


# 1) SFT
#${PY} run_experiments.py --task train --mode sft --model ${MODEL} --dataset ${DATASET} \
#  > "${LOG_DIR}/sft.log" 2>&1
# infer (SFT)
#INFER_OUTPUT_NAME="sft_infer.jsonl" ${PY} run_experiments.py --task infer --mode sft --infer-mode basic --model ${MODEL} --dataset ${DATASET} \
#  --model-path ./out/sft_math \
#  > "${LOG_DIR}/sft_infer.log" 2>&1

# 2) LoRA
${PY} run_experiments.py --task train --mode lora --model ${MODEL} --dataset ${DATASET} \
  > "${LOG_DIR}/lora.log" 2>&1
# infer (LoRA adapter path)
INFER_OUTPUT_NAME="lora_infer.jsonl" ${PY} run_experiments.py --task infer --mode lora --infer-mode basic --model ${MODEL} --dataset ${DATASET} \
  --model-path ./out/lora_math \
  > "${LOG_DIR}/lora_infer_basic.log" 2>&1
  
#INFER_OUTPUT_NAME="lora_infer_bon.jsonl" ${PY} run_experiments.py --task infer --mode lora --infer-mode bon --model ${MODEL} --dataset ${DATASET} \
#  --model-path ./out/lora_math \
#  > "${LOG_DIR}/lora_infer.log" 2>&1

# 3) Prompt-tuning
#${PY} run_experiments.py --task train --mode prompt --model ${MODEL} --dataset ${DATASET} \
#  > "${LOG_DIR}/prompt.log" 2>&1
# infer (Prompt adapter path)
#INFER_OUTPUT_NAME="prompt_infer.jsonl" ${PY} run_experiments.py --task infer --mode prompt --infer-mode basic --model ${MODEL} --dataset ${DATASET} \
#  --model-path ./out/prompt_math \
#  > "${LOG_DIR}/prompt_infer.log" 2>&1
