import torch
import gc
import os

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm

# Load env var
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def eval(model, tokenizer, dataset, batch_size=4):
    device = model.device
    predictions = []
    references = []
        
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        # Get batch
        batch = dataset[i : i + batch_size]
        quotes = batch['quote']
        authors = batch['author']
        
        # Prepare inputs
        prompts = [f"Quote: {q}" for q in quotes]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        # Generate
        with torch.inference_mode():
            # Generate text
            outputs = model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, max_new_tokens=100)
        
        # Decode and score
        outputs = outputs.cpu()
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for j, generated_text in enumerate(generated_texts):
            prompt = prompts[j]
            real_suffix = f"Author: {authors[j]}"
            
            # Remove prompt from generated text
            if generated_text.startswith(prompt):
                generated_suffix = generated_text[len(prompt):].strip()
            else:
                generated_suffix = generated_text.strip()
                
            references.append(real_suffix)
            predictions.append(generated_suffix)
                
            print(f"**********************************")
            print(prompt)
            print(real_suffix)
            print(generated_suffix)

        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()

    return predictions, references

def compute_metrics(predictions, references):
    # Load metrics
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    
    # Calculate ROUGE (Word overlap)
    rouge_results = rouge.compute(predictions=predictions, references=references)

    # Calculate BERT (Semantic similarity)
    bert_results = bertscore.compute(predictions=predictions, references=references, lang="en")
    bert_precision = sum(bert_results['precision']) / len(bert_results['precision'])
    bert_recall = sum(bert_results['recall']) / len(bert_results['recall'])
    bert_f1 = sum(bert_results['f1']) / len(bert_results['f1'])
    
    return {**rouge_results, "bert_precision": bert_precision, "bert_recall": bert_recall, "bert_f1": bert_f1}

# Load the GG model
model_id = "google/gemma-2b" # "TinyLlama/TinyLlama_v1.1"
output_dir = "outputs/gemma-2b-lora/checkpoint-100" # "outputs/tinyllama-v1.1-lora/checkpoint-100"
dataset_name = "Abirate/english_quotes"
quantization_type = "loftq"

dataset = load_dataset(dataset_name, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Quantization config
if quantization_type == "qlora":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=os.environ["HF_TOKEN"],
    device_map={"":0},
    torch_dtype=torch.float16 if "TinyLlama" in model_id else torch.float32,
)
model = PeftModel.from_pretrained(base_model, output_dir, is_trainable=False)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Get predictions and references
train_predictions, train_references = eval(model, tokenizer, train_dataset)
eval_predictions, eval_references = eval(model, tokenizer, eval_dataset)

# Free memory to calculate using rouoge and bert
# Delete the model reference
del model
del base_model
# Force garbage collection
gc.collect()
torch.cuda.empty_cache()

# eval metrics
import evaluate

train_results = compute_metrics(train_predictions, train_references)
print("Train:")
for metric_name in train_results:
    print(f"\t{metric_name}: {train_results[metric_name]:.4f}")

eval_results = compute_metrics(eval_predictions, eval_references)
print("Eval:")
for metric_name in eval_results:
    print(f"\t{metric_name}: {eval_results[metric_name]:.4f}")