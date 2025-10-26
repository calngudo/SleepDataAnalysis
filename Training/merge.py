from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

print("Loading fine-tuned model with adapters...")
model = AutoPeftModelForCausalLM.from_pretrained(
    "gemma-sleep-recommender",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

print("Merging adapters into base model...")
model = model.merge_and_unload()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("./basegemma")

print("Saving merged model...")
model.save_pretrained("gemma-sleep-merged")
tokenizer.save_pretrained("gemma-sleep-merged")

print("✅ Merged model saved to gemma-sleep-merged/")
