""" 
------ After running "filter_code_comment_pairs.py" script --------
This script samples 100 methods per repos and if possible then each method from different files of the repo
"""

import json
import os
import random
from collections import defaultdict
from tqdm import tqdm

input_path = "./data_files/new_unsampled_cleaned_code_comment_pairs_CAT.jsonl"
output_path = "./data_files/new_sampled_code_comment_pairs.jsonl"

# Step 1: Load all entries
entries = []
with open(input_path, 'r') as f:
    for line in f:
        entry = json.loads(line)
        entries.append(entry)

# Step 2: Organize entries by directory and file
dir_to_files = defaultdict(lambda: defaultdict(list))

for entry in entries:
    full_path = entry['file']
    relative_path = full_path.split("python-benchmark-repos", 1)[-1]
    relative_path = relative_path.strip(os.sep)
    top_repo = relative_path.split(os.sep)[0]  # repo name
    file_name = os.path.basename(relative_path)
    dir_to_files[top_repo][file_name].append(entry)

# Step 3: Sample up to 5 files per directory, and 1 code from each
sampled = []

for repo, files_dict in tqdm(dir_to_files.items(), desc="Sampling"):
    selected_files = random.sample(list(files_dict.keys()), min(300, len(files_dict)))
    for fname in selected_files:
        candidates = files_dict[fname]
        sampled_entry = random.choice(candidates)
        sampled.append(sampled_entry)

# Step 4: Write output
with open(output_path, 'w') as out:
    for item in sampled:
        out.write(json.dumps(item) + '\n')

print(f"Sampled {len(sampled)} entries written to {output_path}")