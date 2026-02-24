from dataclasses import dataclass
from typing import Optional
import re
from datasets import load_dataset, Dataset, DatasetDict


@dataclass
class MATHDatasetBuilder:
    train_path: str = "dataset/math/math_train.parquet"
    test_path: str  = "dataset/math/math_test.parquet"

    val_ratio: float = 0.2
    seed: int = 42

    prompt_template: str = (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e."
        ", <think> reasoning process here </think> <answer> answer here </answer>."
        "User: {question}"
        "Assistant: <think>"
    )

    def load(self) -> tuple[Dataset, Dataset]:
        train_ds = load_dataset("parquet", data_files=self.train_path, split="train")
        test_ds  = load_dataset("parquet", data_files=self.test_path,  split="train")
        return train_ds, test_ds
    
    @staticmethod
    def extract_gold_from_solution(solution: str) -> Optional[str]:
        if not solution:
            return None

        s = solution
        key = r"\boxed{"
        start = s.find(key)
        if start == -1:
            return None

        i = start + len(key) 
        depth = 1
        out = []

        while i < len(s):
            ch = s[i]
            if ch == "{":
                depth += 1
                out.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
                out.append(ch)
            else:
                out.append(ch)
            i += 1

        gold = "".join(out).strip()
        return gold.replace(" ", "") if gold else None

    @staticmethod
    def strip_answer_from_solution(solution: str, gold: Optional[str]) -> str:
        if not solution:
            return ""
        s = solution
        s = re.sub(r"\\boxed\{[^}]*\}", "", s)
        if gold:
            parts = s.rsplit(gold, 1)
            s = "".join(parts)
        return s.strip()

    def build_fields(self, ds: Dataset) -> Dataset:
        template = self.prompt_template

        def _map(ex):
            q = ex["problem"]
            a = ex["solution"]
            prompt = template.format(question=q)
            gold = self.extract_gold_from_solution(a)
            return {
                "prompt": prompt,
                "question": q,
                "completion": a + "</think>\n<answer>" + f"{gold}</answer>",
                "gold": gold,
            }


        return ds.select_columns(["problem", "solution"]).map(
            _map,
            remove_columns=["problem", "solution"],
        )

    def build(self) -> DatasetDict:
        train_ds, test_ds = self.load()

        train_ds = self.build_fields(train_ds)
        test_ds  = self.build_fields(test_ds)

        split = train_ds.train_test_split(
            test_size=self.val_ratio,
            seed=self.seed,
        )

        return DatasetDict({
            "train": split["train"],
            "val":   split["test"],
            "test":  test_ds,
        })

def get_math_ds(
    train_path: str = "dataset/math/math_train.parquet",
    test_path: str  = "dataset/math/math_test.parquet",
    val_ratio: float = 0.20,
    seed: int = 42,
    prompt_template: str = (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e."
        ", <think> reasoning process here </think> <answer> answer here </answer>."
        "User: {question}"
        "Assistant: <think>"
    ),
) -> DatasetDict:
    builder = MATHDatasetBuilder(
        train_path=train_path,
        test_path=test_path,
        val_ratio=val_ratio,
        seed=seed,
        prompt_template=prompt_template,
    )
    return builder.build()


def main() -> None:
    builder = MATHDatasetBuilder(
    train_path="dataset/math/math_train.parquet",
    test_path="dataset/math/math_test.parquet",
    val_ratio=0.20,
    )

    ds = builder.build()
    train_ds = ds["train"]
    val_ds = ds["val"]
    test_ds = ds["test"]

    print(ds)
    print("features:", train_ds.features)
    print("test sample gold:", test_ds[0]["gold"])
    print("test sample prompt:", test_ds[0]["prompt"])
    print("test sample completion:", test_ds[0]["completion"])


if __name__ == "__main__":
    main()
