"""
---------- CAT cleaning --------
After running "sample_n-methods_per_dir.py" script tp pick n-samples per repo,
This script apply CAT cleaning tool on every <code, comment> pair to filter the clean code
"""

import sys
import json
import os
from tqdm import tqdm

# Add CAT module path
sys.path.append("/home/user/projects/prompt-project/SIDE_p/scripts/FSE22_BuiltOntheRock/CAT")

# Suppress tqdm output from RuleCleaner by redirecting stderr temporarily
from contextlib import redirect_stderr
from io import StringIO

from rule_cleaner import RuleCleaner

# === File paths ===
input_path = "./data_files/new_python_code_comment_pairs_filtered.jsonl"
output_path = "./data_files/new_unsampled_cleaned_code_comment_pairs_CAT.jsonl"

# === Process entries one by one ===
kept = 0
skipped_empty = 0
skipped_noisy = 0
skipped_short = 0

# Track noise statistics
noise_stats = {
    'ContentTamper': 0,
    'NonLiteral': 0,
    'Interrogation': 0,
    'UnderDevelop': 0,
    'EmptyFunc': 0,
    'CommentOut': 0,
    'BlockComment': 0,
    'AutoCode': 0,
    'DuplicatedCode': 0
}

print("Processing entries one by one...")

with open(input_path, "r") as infile, open(output_path, "w") as outfile:
    for line_num, line in enumerate(tqdm(infile, desc="Cleaning entries")):
        try:
            data = json.loads(line)
            raw_code = data.get("code", "")
            raw_comment = data.get("docstring", "")
            
            # Skip entries with empty comments
            if not raw_comment or raw_comment.strip() == "":
                skipped_empty += 1
                continue
            
            # Apply CAT cleaning to this single entry (suppress tqdm output)
            try:
                with redirect_stderr(StringIO()):
                    cleaner = RuleCleaner([raw_code], [raw_comment])
                    cleaned_code_list, cleaned_comment_list = cleaner.get_clean_data()
                
                # Get noise statistics for this entry
                try:
                    noisy_data = cleaner.get_noisy_data()
                    for noise_type, noisy_entries in noisy_data.items():
                        if noise_type in noise_stats:
                            noise_stats[noise_type] += len(noisy_entries)
                except:
                    pass  # Continue if noise stats not available
                
                # Check if the entry survived CAT cleaning
                if len(cleaned_code_list) == 0 or len(cleaned_comment_list) == 0:
                    skipped_noisy += 1
                    continue
                
                # Get the cleaned versions
                clean_code = cleaned_code_list[0]
                clean_comment = cleaned_comment_list[0].strip()
                
                # Apply additional filtering for very short comments
                if not clean_comment or len(clean_comment.split()) < 3:
                    skipped_short += 1
                    continue
                
                # Create output entry with original data plus cleaned versions
                output_data = data.copy()
                output_data["clean_code"] = clean_code
                output_data["clean_docstring"] = clean_comment
                
                # Write to output file
                outfile.write(json.dumps(output_data) + "\n")
                kept += 1
                
            except Exception as e:
                print(f"⚠️ Error processing entry {line_num}: {e}")
                skipped_noisy += 1
                continue
                
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error on line {line_num}: {e}")
            continue

print(f"\n✅ Results:")
print(f"   Total processed entries: {line_num + 1}")
print(f"   Kept entries: {kept}")
print(f"   Skipped (empty comments): {skipped_empty}")
print(f"   Skipped (detected as noisy): {skipped_noisy}")
print(f"   Skipped (too short after cleaning): {skipped_short}")
print(f"   Total skipped: {skipped_empty + skipped_noisy + skipped_short}")
print(f"   Output written to: {output_path}")

# === Display noise statistics ===
print(f"\n Noise Statistics:")
total_noisy_detected = 0
for noise_type, count in noise_stats.items():
    print(f"   {noise_type}: {count} entries")
    total_noisy_detected += count
print(f"   Total noisy entries detected: {total_noisy_detected}")

# === Optional: Quick validation ===
'''if kept > 0:
    print(f"\n Quick validation:")
    with open(output_path, "r") as f:
        first_line = f.readline()
        sample_entry = json.loads(first_line)
        required_fields = ["file", "function_name", "code", "docstring", "clean_code", "clean_docstring"]
        missing_fields = [field for field in required_fields if field not in sample_entry]
        if missing_fields:
            print(f"   ⚠️ Missing fields in output: {missing_fields}")
        else:
            print(f"   ✅ All required fields present in output")
        print(f"   ✅ Sample clean_docstring length: {len(sample_entry['clean_docstring'].split())} words")
'''