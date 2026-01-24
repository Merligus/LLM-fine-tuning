import torch
import os

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm

# eval metrics
from Levenshtein import ratio


load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

def formatting_func(example):
    text = f"### Quote: {example['quote']}\n### Author: {example['author']}"
    return text

def eval(model, tokenizer, dataset, batch_size=16):
    device = model.device
    total_score = 0
    
    # Ensure pad token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        # Get batch
        batch = dataset[i : i + batch_size]
        quotes = batch['quote']
        authors = batch['author']
        
        # Prepare inputs
        prompts = [f"### Quote: {q}" for q in quotes]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, pad_token_id=tokenizer.pad_token_id)
        
        # Decode and score
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for j, generated_text in enumerate(generated_texts):
            prompt = prompts[j]
            real_suffix = f"### Author: {authors[j]}"
            
            # Remove prompt from generated text
            if generated_text.startswith(prompt):
                generated_suffix = generated_text[len(prompt):].strip()
            else:
                generated_suffix = generated_text.strip()
            score = ratio(real_suffix, generated_suffix)
            print(f"**********************************")
            print(prompt)
            print(real_suffix)
            print(generated_suffix)
            print(score)
            total_score += score
            
    return total_score / len(dataset)


# Load the GG model
model_id = "google/gemma-2b"
output_dir = "outputs/gemma-lora/checkpoint-10"
dataset_name = "Abirate/english_quotes"

dataset = load_dataset(dataset_name, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    token=os.environ["HF_TOKEN"],
    device_map={"":0},
)
model = PeftModel.from_pretrained(base_model, output_dir)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

train_score = 0 # eval(model, tokenizer, train_dataset)
eval_score = eval(model, tokenizer, eval_dataset)
print(f"Train score: {train_score * 100:.2f}\nEval score: {eval_score * 100:.2f}")