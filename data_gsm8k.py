from datasets import load_dataset, DatasetDict
from typing import Optional
from dataclasses import dataclass
import re

@dataclass
class GSM8KDatasetBuilder:
    train_path: str = "dataset/gsm8k/gsm8k_train.parquet"
    test_path: str  = "dataset/gsm8k/gsm8k_test.parquet"
    
    val_ratio: float = 0.20
    seed: int = 42

    split_train: str = "train"
    split_test: str = "test"
    
    prompt_template: str = (
        "Please solve the following math problem step by step. \n\n"
        "Question: {question}\n\n"
        "Solution:\n"
    )

    def load(self) -> DatasetDict:
        ds = load_dataset(
            "parquet",
            data_files={
                self.split_train: self.train_path,
                self.split_test: self.test_path,
            },
        )
        return ds
    
    @staticmethod
    def extract_gold(answer: str) -> Optional[str]:
        m = re.search(r"####\s*([^\n]+)", answer or "")
        if not m:
            return None
        s = m.group(1).strip().replace(",", "")
        m2 = re.search(r"-?\d+(?:\.\d+)?", s)
        return m2.group(0) if m2 else s
    
    def build_fields(self, ds: DatasetDict) -> DatasetDict:
        template = self.prompt_template

        def _map(ex):
            q = ex["question"]
            a = ex["answer"]
            prompt = template.format(question=q)
            gold = self.extract_gold(a)
            # a = "\n".join([line for line in a.splitlines() if not line.strip().startswith("####")]).strip()
            return {
                "prompt": prompt,
                "question": q,
                "completion": a,
                # "text": prompt + a,
                "gold": gold,
            }

        remove_cols = ds["train"].column_names
        return ds.map(_map, remove_columns=remove_cols)

    def build(self) -> DatasetDict:
        raw = self.load()
        processed = self.build_fields(raw)

        split = processed["train"].train_test_split(
            test_size=self.val_ratio,
            seed=self.seed,
        )

        return DatasetDict({
            "train": split["train"],
            "val":   split["test"],  
            "test":  processed["test"]
        })

def get_gsm8k_ds(
    train_path: str = "dataset/gsm8k/gsm8k_train.parquet",
    test_path: str  = "dataset/gsm8k/gsm8k_test.parquet",
    val_ratio: float = 0.20,
    seed: int = 42,
    prompt_template: str = (
        "Please solve the following math problem step by step.\n\n"
        "Question: {question}\n\n"
        "Solution:\n"
    ),
) -> DatasetDict:
    builder = GSM8KDatasetBuilder(
        train_path=train_path,
        test_path=test_path,
        val_ratio=val_ratio,
        seed=seed,
        prompt_template=prompt_template,
    )
    return builder.build()


def main() -> None:
    builder = GSM8KDatasetBuilder(
    train_path="dataset/gsm8k/gsm8k_train.parquet",
    test_path="dataset/gsm8k/gsm8k_test.parquet",
    val_ratio=0.20,
    )

    ds = builder.build()
    train_ds = ds["train"]
    val_ds = ds["val"]
    test_ds = ds["test"]

    print(ds)
    print("features:", train_ds.features)
    print("train sample gold:", train_ds[0]["gold"])
    print("train sample text:", train_ds[0]["text"][:1024])


if __name__ == "__main__":
    main()
