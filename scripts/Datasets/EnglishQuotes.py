from datasets import load_dataset


class EnglishQuotesDataset:
    def __init__(self, dataset_name, bos_token, eos_token):
        # Store parameters
        self.dataset_name = dataset_name
        self.eos_token = eos_token
        self.bos_token = bos_token

        # Load dataset
        dataset = load_dataset(self.dataset_name, split="train")
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        self.train_dataset = dataset["train"]
        self.eval_dataset = dataset["test"]

    def formatting_func(self, example):
        text = (
            self.bos_token
            + f"Quote: {example['quote']}\nAuthor: {example['author']}"
            + self.eos_token
        )
        print(text)
        return text
