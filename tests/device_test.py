import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"

# Check GPU details
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device detected: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Test generation
inputs = tokenizer("Hello world!", return_tensors="pt").to(device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=5)

print("Generated tokens:", tokenizer.decode(output[0]))
print("Model device:", next(model.parameters()).device)

# Performance test
import time
start = time.time()
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=20)
end = time.time()
print(f"Generation time: {(end - start)*1000:.2f} ms")