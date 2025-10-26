import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from trl import SFTConfig


import os

# --- 1. Your Training Data ---
training_data = [
    {
        "text": """<start_of_turn>user
A user's weekly health summary is as follows:
* Average Sleep Score: 69 (Fair)
* Average Sleep Duration: 6 hours 15 minutes
* Average Stress Level: 36
* Average Daily Steps: 11,623
* Average Bedtime: 1:22 AM

Based on this data, what is your sleep recommendation?<start_of_turn>model
Your sleep score is 'Fair.' Your step count is excellent, which is great for overall health! However, your average sleep duration of 6h 15m is below the recommended 7-8 hours. Your late bedtime of 1:22 AM is the most likely cause.

**Recommendation:** Try to set a consistent "wind-down" alarm for 12:15 AM to begin preparing for sleep. Aiming for a bedtime closer to 12:45 AM would help you get over 7 hours of sleep.
<end_of_turn>"""
    },
    {
        "text": """<start_of_turn>user
A user's weekly health summary is as follows:
* Average Sleep Score: 67 (Fair)
* Average Sleep Duration: 5 hours 52 minutes
* Average Stress Level: 38
* Average Daily Steps: 20,351
* Average Bedtime: 12:26 AM

Based on this data, what is your sleep recommendation?<start_of_turn>model
Your 'Fair' sleep score is directly linked to your average duration of 5h 52m, which is significantly under the 7-hour minimum. While your high step count is fantastic, your body needs time to recover from that high activity, and it isn't getting it. Your stress is also slightly elevated.

**Recommendation:** Your 12:26 AM bedtime is good, but you may be waking up too early. If you can't sleep later, you must shift your bedtime earlier. Aim for a 11:45 PM bedtime to push your duration over 6.5 hours.
<end_of_turn>"""
    },
    {
        "text": """<start_of_turn>user
A user's weekly health summary is as follows:
* Average Sleep Score: 48 (Poor)
* Average Sleep Duration: 4 hours 8 minutes
* Average Stress Level: 38
* Average Daily Steps: 20,464
* Average Bedtime: 2:51 AM

Based on this data, what is your sleep recommendation?<start_of_turn>model
Your sleep score is 'Poor,' and your average duration of 4h 8m is a major health concern. This level of sleep deprivation can seriously impact your mood, energy, and long-term health, and is likely contributing to your stress level of 38. Your very late bedtime is the clear problem.

**Recommendation:** A 2:51 AM bedtime is not sustainable. You must prioritize sleep. Start by setting a firm bedtime of 1:30 AM and gradually pull it earlier by 15 minutes every few days. Do not compromise on this.
<end_of_turn>"""
    },
    {
        "text": """<start_of_turn>user
A user's weekly health summary is as follows:
* Average Sleep Score: 73 (Fair)
* Average Sleep Duration: 6 hours 50 minutes
* Average Stress Level: 35
* Average Daily Steps: 30,656
* Average Bedtime: 12:43 AM

Based on this data, what is your sleep recommendation?<start_of_turn>model
You are very close to a 'Good' sleep score! Your 6h 50m duration is just shy of the 7-hour mark. Your activity level is extremely high (30k+ steps!), which is incredible. Your 12:43 AM bedtime is reasonable.

**Recommendation:** Just 10-15 more minutes of sleep per night would make a big difference. Try to be in bed by 12:30 AM instead of 12:43 AM. This small change is all you need to move into the 'Good' category.
<end_of_turn>"""
    },
    {
        "text": """<start_of_turn>user
A user's weekly health summary is as follows:
* Average Sleep Score: 65 (Fair)
* Average Sleep Duration: 6 hours 21 minutes
* Average Stress Level: 40
* Average Daily Steps: 29,369
* Average Bedtime: 12:58 AM

Based on this data, what is your sleep recommendation?<start_of_turn>model
Your 'Fair' score is a result of sleep duration under 6.5 hours and a high-stress level of 40. Your high step count is great, but combined with high stress and low sleep, you are at risk for burnout. Your 12:58 AM bedtime is pushing late.

**Recommendation:** Focus on stress reduction before bed. Avoid screens for 30 minutes before your target bedtime. Try to move your bedtime to 12:30 AM to get at least 7 hours of sleep, which will help your body recover from your high activity and manage stress.
<end_of_turn>"""
    }
]

# Convert to Dataset
dataset = Dataset.from_list(training_data)

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
