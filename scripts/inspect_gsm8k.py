"""Read and inspect GSM8K answer formatting. Needs pip install -e '.[data]'"""
from datasets import load_dataset
from grpo_ablation.rewards import gsm8k_ground_truth


def main():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for sample in ds.select(range(3)):
        print("Q:", sample["question"])
        print("GT:", gsm8k_ground_truth(sample["answer"]))
        print("-" * 60)


if __name__ == "__main__":
    main()
