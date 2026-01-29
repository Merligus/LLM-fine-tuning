from dataclasses import dataclass, field
from typing import Optional

import torch
import os

from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, LoftQConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from Datasets.EnglishQuotes import EnglishQuotesDataset

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)
    max_seq_length: Optional[int] = field(default=2048)
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."},
    )
    dataset_name: Optional[str] = field(
        default="Abirate/english_quotes",
        metadata={"help": "The preference dataset to use."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash Attention 2."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    quantization_type: str = field(
        default="qlora",
        metadata={"help": "Quantization type choosing between LoftQ and QLora or None. Possible values loftq, qlora, none"},
    )
    ft_method: str = field(
        default="dora",
        metadata={"help": "Fine tuning method to train the model choosing between Lora and Dora. Possible values lora, dora"},
    )
    max_steps: int = field(default=100, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    save_steps: int = field(default=10, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=1, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(
        default="./results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


# Load the GG model
model_id = "meta-llama/Llama-3.2-1B"  # "TinyLlama/TinyLlama_v1.1", "google/gemma-2b"
output_dir = "outputs/llama-3.2-lora"  # "outputs/tinyllama-v1.1-lora", "outputs/gemma-2b-lora"

# Additional parameters
additional_lora_configs = {}
quantization_config = None

if script_args.ft_method == "dora":
    additional_lora_configs["use_dora"] = True

if script_args.quantization_type == "qlora":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=(torch.bfloat16 if "TinyLlama" in model_id else torch.float16),
        bnb_4bit_quant_type="nf4",
    )
elif script_args.quantization_type == "loftq":
    additional_lora_configs["init_lora_weights"] = "loftq"
    additional_lora_configs["loftq_config"] = LoftQConfig(loftq_bits=4)

# Load model
print("Load model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation=("sdpa" if not script_args.use_flash_attention_2 else "flash_attention_2"),
    token=os.environ["HF_TOKEN"],
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16 if "TinyLlama" in model_id else torch.float32,
    device_map="auto" if quantization_config is not None else None,
)

# Casts LayerNorms to float32 and prepares the model for gradient checkpointing
# Avoid Inf NaN errors
if quantization_config is not None:
    model = prepare_model_for_kbit_training(model)

# Load tokenizer
print("Load tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Update model config to match tokenizer
model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

lora_config = LoraConfig(
    r=script_args.lora_r,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    # LoftQ and Dora configs
    **additional_lora_configs,
)

if script_args.dataset_name == "Abirate/english_quotes":
    dataset = EnglishQuotesDataset(script_args.dataset_name, tokenizer.bos_token, tokenizer.eos_token)

training_arguments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_ratio=script_args.warmup_ratio,
    lr_scheduler_type=script_args.lr_scheduler_type,
    gradient_checkpointing=script_args.gradient_checkpointing,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    packing=script_args.packing,
    max_length=script_args.max_seq_length,
)

print("Start training...")
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_arguments,
    train_dataset=dataset.train_dataset,
    eval_dataset=dataset.eval_dataset,
    peft_config=lora_config,
    formatting_func=dataset.formatting_func,
)

trainer.train()

print(f"{script_args.ft_method} x {script_args.quantization_type} training finished")
