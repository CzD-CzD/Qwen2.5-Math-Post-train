import json, re, sys

def extract_num(text: str):
    if text is None:
        return None
    text = re.sub(r"(?im)^human:\s*", "", text)
    text = re.sub(r"(?im)^step\s*\d+\s*:\s*", "", text)
    m = re.search(r"####\s*([^\n]+)", text)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None



def eval_file(path: str):
    total = 0
    correct = 0
    for line in open(path, "r", encoding="utf-8"):
        if not line.strip():
            continue
        obj = json.loads(line)
        pred = extract_num(obj.get("prediction", ""))
        gold = extract_num(obj.get("gold", ""))
        if pred is not None and gold is not None:
            total += 1
            if pred == gold:
                correct += 1
    acc = correct / total if total else 0
    print(f"{path} | total={total} correct={correct} acc={acc:.4f}")

if __name__ == "__main__":
    eval_file(sys.argv[1])

# python eval_gsm8k.py infer_outputs/base_infer.jsonl 