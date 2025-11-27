import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from trl import SFTConfig
import os

#Dataset Loader
dataset = load_dataset("json", data_files="dataset3.txt", split="train")
#Base Model Loader
model_id = "./gemma2-2b"


#Training Settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Loading model from {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,  #Do not attempt to download from GitHub
)
model.config.use_cache = False


tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    local_files_only=True  
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)


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

print("Starting training...")
trainer.train()
print("Training complete.")


trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")
