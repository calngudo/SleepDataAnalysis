import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from trl import SFTConfig


import os

# --- 1. Your Training Data ---
# This section has been changed to load data from the text file.
# The original 'training_data' list has been removed.
dataset = load_dataset("json", data_files="dataset.txt", split="train")


# --- 2. Load Model & Tokenizer from LOCAL PATH ---

# CHANGE THIS to your local folder path
model_id = "./basegemma"  # <-- Local path instead of "google/gemma-2b-it"

# QLoRA configuration (4-bit quantization)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load base model from local folder
print(f"Loading model from {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,  # <-- This ensures it doesn't try to download
)
model.config.use_cache = False

# Load tokenizer from local folder
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    local_files_only=True  # <-- This ensures it doesn't try to download
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 3. Configure LoRA ---
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# --- 4. Configure Training ---
output_dir = "gemma-sleep-recommender"
sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    dataset_text_field="text",
 
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=sft_config,
)
# --- 6. Start Training ---
print("Starting training...")
trainer.train()
print("Training complete!")

# --- 7. Save Model ---
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")
