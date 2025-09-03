"""
---------- CAT cleaning --------
After running "sample_n-methods_per_dir.py" script tp pick n-samples per repo,
This script apply CAT cleaning tool on every <code, comment> pair to filter the clean code
"""

import sys
import json
from tqdm import tqdm

# Add CAT module path
sys.path.append("/home/user/projects/prompt-project/SIDE_p/scripts_and_data/FSE22_BuiltOntheRock/CAT")

from rule_cleaner import RuleCleaner

# === File paths ===
input_path = "/home/user/projects/prompt-project/codet5+/dataset/funcom-python/Funcom-data-AST/funcom_train_AST_python_RAW.jsonl"
output_path = "/home/user/projects/prompt-project/codet5+/dataset/funcom-python/Funcom_python_without_SIDE/funcom_python_train_CAT-cleaned.jsonl"

# === Load data and prepare for CAT ===
raw_code_list = []
raw_comment_list = []
original_data = []

print("Loading data...")
with open(input_path, "r") as infile:
    for line in infile:
        data = json.loads(line)
        #raw_code = data.get("code", "")
        tokens = data.get("code_tokens", "")
        raw_code = " ".join(tokens)
        raw_comment = data.get("docstring", "")
        #raw_code = data.get("clean_code", "")
        #raw_comment = data.get("clean_docstring", "")
        if not raw_comment or raw_comment.strip() == "":
            continue  # Skip empty/null comments
        raw_code_list.append(raw_code)
        raw_comment_list.append(raw_comment)
        original_data.append(data)

print(f"Loaded {len(original_data)} entries for cleaning")

# === Apply CAT Cleaning ===
print("Applying CAT cleaning...")
try:
    cleaner = RuleCleaner(raw_code_list, raw_comment_list)
    cleaned_code_list, cleaned_comment_list = cleaner.get_clean_data()
    print(f"CAT cleaning completed: {len(original_data)} -> {len(cleaned_code_list)} entries")
except Exception as e:
    print("❌ Error during CAT cleaning:", e)
    sys.exit(1)

# === Create mapping between original and cleaned data ===
# Since CAT removes entries, we need to figure out which original entries
# correspond to the cleaned ones. We'll do this by matching content.
print("Creating mapping between original and cleaned data...")

cleaned_to_original_map = []
used_indices = set()

for i, (clean_code, clean_comment) in enumerate(zip(cleaned_code_list, cleaned_comment_list)):
    # Find the matching original entry
    best_match_idx = None
    
    # First, try exact match on docstring (most reliable)
    for j, orig_data in enumerate(original_data):
        if j in used_indices:
            continue
        #if orig_data["clean_docstring"].strip() == clean_comment.strip():
        if orig_data["docstring"].strip() == clean_comment.strip():
            best_match_idx = j
            break
    
    # If no exact docstring match, try to find by code similarity
    if best_match_idx is None:
        for j, orig_data in enumerate(original_data):
            if j in used_indices:
                continue
            # Check if the cleaned code is a subset/version of original code
            if clean_code.strip() in orig_data["code"] or orig_data["code"] in clean_code.strip():
            #if clean_code.strip() in orig_data["clean_code"] or orig_data["clean_code"] in clean_code.strip():
                best_match_idx = j
                break
    
    # Last resort: take the next unused index (assumes some order preservation)
    if best_match_idx is None:
        for j in range(len(original_data)):
            if j not in used_indices:
                best_match_idx = j
                break
    
    if best_match_idx is not None:
        cleaned_to_original_map.append(best_match_idx)
        used_indices.add(best_match_idx)
    else:
        print(f"⚠️ Warning: Could not find original data for cleaned entry {i}")
        cleaned_to_original_map.append(None)

# === Write Cleaned Output ===
kept = 0
skipped = 0

print("Writing cleaned output...")
with open(output_path, "w") as outfile:
    for i, original_idx in enumerate(cleaned_to_original_map):
        if original_idx is None:
            skipped += 1
            continue
            
        clean_comment = cleaned_comment_list[i].strip()
        clean_code = cleaned_code_list[i]
        
        # Apply additional filtering if needed
        if clean_comment and len(clean_comment.split()) >= 3:
            # Create output entry with original data plus cleaned versions
            output_data = original_data[original_idx].copy()
            output_data["clean_code"] = clean_code
            output_data["clean_docstring"] = clean_comment
            
            outfile.write(json.dumps(output_data) + "\n")
            kept += 1
        else:
            skipped += 1

total_removed = len(original_data) - kept

print(f"✅ Results:")
print(f"   Original entries: {len(original_data)}")
print(f"   Entries after CAT cleaning: {len(cleaned_code_list)}")
print(f"   Final kept entries: {kept}")
print(f"   Skipped entries: {skipped}")
print(f"   Total removed: {total_removed}")
print(f"   Output written to: {output_path}")

# === Optional: Get and display noisy data statistics ===
try:
    noisy_data = cleaner.get_noisy_data()
    print(f"\n📊 Noise Statistics:")
    total_noisy = 0
    for noise_type, noisy_entries in noisy_data.items():
        count = len(noisy_entries)
        total_noisy += count
        print(f"   {noise_type}: {count} entries")
    print(f"   Total noisy entries detected: {total_noisy}")
except Exception as e:
    print(" Noise statistics not available:", e)