from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv

# Load env var
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_id = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Update model config to match tokenizer
model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False


# Simple "G-Eval" style prompt using a local LLM
def evaluate_with_llm(predictions, references):
    prompts = [
        f"""
        You are a strict judge. Compare the Prediction to the Reference.
        
        Rules:
        1. If the Author is wrong, score is 0.
        2. If the Quote is slightly off but meaning is kept, score is 0.8.
        3. If perfect, score is 1.0.
        
        Reference: {r}
        Prediction: {p}
        
        Output score is
        """
        for p, r in zip(predictions, references)
    ]
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

    # Generate
    with torch.inference_mode():
        # Generate text
        outputs = model.generate(**input_ids, pad_token_id=tokenizer.pad_token_id, max_new_tokens=10)

    # Decode and score
    outputs = outputs.cpu()
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    scores = []
    for j, generated_text in enumerate(generated_texts):
        prompt = prompts[j]

        # Remove prompt from generated text
        if generated_text.startswith(prompt):
            generated_suffix = generated_text[len(prompt) :].strip()
        else:
            generated_suffix = generated_text.strip()

        try:
            score = float("".join(c for c in generated_suffix if c.isdigit() or c == "."))
            scores.append(score)
        except (ValueError, IndexError):
            print(f"ValueError/IndexError for {j} prompt")
            scores.append(0.0)

    return scores


predictions = ["abuble hahaha", "Bram Stoker", "Author: Marcos Paulo", "", "Author: Bram Stoker"]
references = [
    "Author: Bram Stoker",
    "Author: Bram Stoker",
    "Author: Bram Stoker",
    "Author: Bram Stoker",
    "Author: Bram Stoker",
]
print(evaluate_with_llm(predictions, references))
