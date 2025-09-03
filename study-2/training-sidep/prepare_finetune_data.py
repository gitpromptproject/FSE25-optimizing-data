import json
import random
import re
from pathlib import Path
from typing import List

# ========== Load Dataset ==========
input_path = "/home/user/projects/prompt-project/CodeXGLUE/Code-Text/code-to-text/dataset/python/train.jsonl"  
output_path = "side_finetune_codexglue_train-new.json"      

with open(input_path, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

# ========== Helper Functions ==========

def extract_inline_comments(code: str) -> List[str]:
    """Extract inline comments that start with '#' but are not full-line comments."""
    return [line.split('#', 1)[1].strip()
            for line in code.split('\n')
            if '#' in line and not line.strip().startswith('#')]

def count_code_lines(code: str) -> int:
    """Count non-empty, non-comment lines of code."""
    return len([line for line in code.splitlines()
                if line.strip() and not line.strip().startswith('#')])

# ========== Convert to SIDE Format ==========
random.seed(42)
side_data = []
hard_negative_count = 0

for i, entry in enumerate(dataset):
    code_tokens = entry["code_tokens"]
    code = " ".join(code_tokens)
    docstring_tokens = entry["docstring_tokens"]
    pos = " ".join(docstring_tokens)
    print(code, pos)
    #pos = entry["docstring"]

    # Pick a random negative from another entry
    neg_idx = random.choice([j for j in range(len(dataset)) if j != i])
    neg_docstring_tokens = dataset[neg_idx]["docstring_tokens"]
    neg = " ".join(neg_docstring_tokens)
    print("NEG\n", neg)

    # Hard negative generation
    total_code_lines = count_code_lines(code)
    inline_comments = extract_inline_comments(code)

    hard_negatives = []
    for comment in inline_comments:
        if any(bad in comment.lower() for bad in ["todo", "to-do", "fixme", "fix-me," "xxx", "hackme", "hack-me", "debug", "remove"]):
            continue
        if total_code_lines == 0:
            continue
        if (1 / total_code_lines) < 0.25:
            hard_negatives.append(comment)

    if hard_negatives:
        hard_negative_count += 1

    side_data.append({
        "query": code,
        "pos": pos.strip(),
        "neg": neg.strip(),
        "hardNegative": hard_negatives
    })

# ========== Save to File ==========
'''with open(output_path, "w", encoding="utf-8") as f:
    for row in side_data:
        json.dump(row, f)
        f.write("\n")'''
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(side_data, f, indent=4)

print(f"✅ Processed {len(side_data)} examples.")
print(f"✅ Found {hard_negative_count} examples with at least one hard negative.")
print(f"📄 Output saved to: {output_path}")