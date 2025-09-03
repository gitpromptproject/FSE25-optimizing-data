# Script 1: run_inference_codet5.py
# This script loads a fine-tuned CodeT5 model, generates predictions for a test set, and saves the output in a JSON file.

import json
import os
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration

model_dir = "path/to/your/fine-tuned-model"
test_file = "path/to/test.jsonl"
output_file = "predictions_codet5_test.json"

# Load model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")
device = model.device

# Load test data
test_data = []
with open(test_file, 'r') as f:
    for line in f:
        ex = json.loads(line)
        code_input = " ".join(ex["shortened_code"]) if isinstance(ex["shortened_code"], list) else ex["shortened_code"]
        test_data.append({"id": ex["id"], "code_input": code_input, "gold": ex["docstring"]})

# Run inference
results = []
for ex in tqdm(test_data, desc="Generating predictions"):
    input_ids = tokenizer(ex["code_input"], return_tensors="pt", truncation=True, padding=True, max_length=512).input_ids.to(device)
    with torch.no_grad():
        generated_ids = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
    pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    results.append({"id": ex["id"], "raw_predictions": pred, "reference": ex["gold"]})

# Save output
with open(output_file, 'w') as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print(f"Predictions saved to {output_file}")


