import json, re, sys
from sympy import simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


def extract_answer(text: str):
    if text is None:
        return None

    matches = re.findall(
        r"</think>\s*<answer>\s*([^<\n]+)\s*</answer>",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return matches[-1].strip()

    m = re.search(r"####\s*([^\n]+)", text)
    if m:
        return m.group(1).strip()

    return text.strip()


def extract_number(text: str):
    if text is None:
        return None
    s = text.replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else None

def normalize_expr(s: str):
    s = s.replace(",", "").strip()

    s = re.sub(r"^\$+", "", s)
    try:
        return parse_latex(s)
    except Exception:
        pass
    try:
        return parse_expr(s)
    except Exception:
        pass
    return s

def is_equal(pred: str, gold: str) -> bool:
    if pred is None or gold is None:
        return False

    pred = pred.strip()
    gold = gold.strip()

    if pred == gold:
        return True

    pred_num = extract_number(pred)
    gold_num = extract_number(gold)
    if pred_num is not None and gold_num is not None and pred_num == gold_num:
        return True

    try:
        p = normalize_expr(pred)
        g = normalize_expr(gold)
        return simplify(p - g) == 0
    except Exception:
        return False


def eval_file(path: str):
    total = 0
    correct = 0
    format_ok = 0
    for line in open(path, "r", encoding="utf-8"):
        if not line.strip():
            continue
        obj = json.loads(line)
        pred_text = obj.get("prediction", "")
        pred = extract_answer(pred_text)
        gold = extract_answer(obj.get("gold", ""))
        total += 1
        if is_equal(pred, gold):
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
