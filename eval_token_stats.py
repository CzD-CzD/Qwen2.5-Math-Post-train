import json
import sys


def eval_file(path: str) -> None:
    total = 0
    sum_prompt = 0
    sum_gen = 0
    sum_total = 0
    missing = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            p = obj.get("prompt_tokens")
            g = obj.get("gen_tokens")
            t = obj.get("total_tokens")
            if p is None or g is None or t is None:
                missing += 1
                continue
            total += 1
            sum_prompt += int(p)
            sum_gen += int(g)
            sum_total += int(t)

    if total == 0:
        print(f"{path} | no token stats found (missing={missing})")
        return

    print(f"{path} | n={total} missing={missing}")
    print(f"prompt_tokens_mean={sum_prompt / total:.2f}")
    print(f"gen_tokens_mean={sum_gen / total:.2f}")
    print(f"total_tokens_mean={sum_total / total:.2f}")


if __name__ == "__main__":
    eval_file(sys.argv[1])
