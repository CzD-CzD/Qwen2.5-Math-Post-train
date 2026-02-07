import json, re, sys

def extract_num(text: str):
    if text is None:
        return None

    m = re.search(r"</think>\s*<answer>\s*([^<\n]+)\s*</answer>", text, flags=re.IGNORECASE)
    if m:
        s = m.group(1).strip().replace(",", "")
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        if nums:
          return nums[-1]

    m = re.search(r"####\s*([^\n]+)", text)
    if m:
        s = m.group(1).strip().replace(",", "")
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        if nums:
          return nums[-1]

    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None





def eval_file(path: str):
    total = 0
    correct = 0
    format_ok = 0
    for line in open(path, "r", encoding="utf-8"):
        if not line.strip():
            continue
        obj = json.loads(line)
        pred_text = obj.get("prediction", "")
        pred = extract_num(obj.get("prediction", ""))
        gold = extract_num(obj.get("gold", ""))
        total += 1
        if pred is not None and pred == gold:
            correct += 1
        if re.search(r"</think>\s*<answer>\s*[^<\n]+\s*</answer>", pred_text, flags=re.IGNORECASE):
            format_ok += 1
            
            
    acc = correct / total if total else 0
    print(f"{path} | total={total} correct={correct} acc={acc:.4f}")
    
    format_acc = format_ok / total if total else 0
    print(f"format_acc={format_acc:.4f}")


if __name__ == "__main__":
    eval_file(sys.argv[1])

# python eval_gsm8k.py infer_outputs/base_infer.jsonl 