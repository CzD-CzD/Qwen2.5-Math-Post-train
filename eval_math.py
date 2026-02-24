import json
import re
import sys

from utils.reward_math import extract_boxed_answer, grade


def extract_answer_relaxed(text: str):
    if text is None:
        return None
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        block = matches[-1].strip()
        boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", block)
        if boxed_matches:
            return boxed_matches[-1].strip()
        boxed = extract_boxed_answer(block)
        if boxed:
            return boxed.strip()
        latex = re.findall(r"\$([^$]+)\$", block, flags=re.DOTALL)
        if latex:
            return latex[-1].strip()
        #nums = re.findall(r"-?\d+(?:\.\d+)?", block.replace(",", ""))
        #if nums:
        #    return nums[-1]
        return block
    boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed_matches:
        return boxed_matches[-1].strip()
    latex = re.findall(r"\$([^$]+)\$", text, flags=re.DOTALL)
    if latex:
        return latex[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if nums:
        return nums[-1]
    return None


def eval_file(path: str) -> None:
    total = 0
    correct = 0
    format_ok = 0

    pattern = re.compile(r"</think>\s*<answer>\s*[^<\n]+\s*</answer>", re.IGNORECASE)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pred_text = obj.get("prediction", "")
            pred = extract_answer_relaxed(pred_text) if pred_text else None
            gold = obj.get("gold")

            total += 1
            if pred is not None and gold is not None and grade(pred, gold, fast=True):
                correct += 1
            if pattern.search(pred_text or ""):
                format_ok += 1

    acc = correct / total if total else 0.0
    format_acc = format_ok / total if total else 0.0

    print(f"{path} | total={total} correct={correct} acc={acc:.4f}")
    print(f"format_acc={format_acc:.4f}")


if __name__ == "__main__":
    eval_file(sys.argv[1])
