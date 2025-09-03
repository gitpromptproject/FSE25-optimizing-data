"""
----- After running "extract_python_methods.py" script ---
This script filters extracted Python functions to retain high-quality and diverse samples
for downstream tasks such as code summarization or generation.

Filter Criteria:
----------------
1. **Remove functions** that match any of the following:
   - Missing or empty docstrings
   - Very short or trivial implementations
   - Auto-generated code (identified by comments like '# auto-generated')

2. **Optional filtering** to ensure diversity and relevance:
   - Function length: retain methods within a range (e.g., 5–20 lines)
   - Docstring length: retain docstrings with 1–3 non-empty lines
   - Docstring quality: discard docstrings with vague phrases like "does stuff", "function", etc.

File Path Normalization:
------------------------
Each function's metadata will include a `file` field storing the relative path from the 
root of the cloned repository. For example:

    /ray/python/ray/autoscaler/v2/scheduler.py

This ensures traceability and reproducibility across filtered datasets.
"""


import json
import os
import re

# === Configuration ===
input_file = "./data_files/new_python_code_comment_pairs.jsonl"
output_file = "./data_files/new_python_code_comment_pairs_filtered.jsonl"
repo_root = "/home/user/projects/prompt-project/SIDE_p/python-benchmark-repos"

# === Helper Functions ===

def is_trivial_docstring(docstring):
    # Check for generic phrases like "does stuff", "function", etc.
    return bool(re.search(r'\b(does stuff|function|helper|perform|do something)\b', docstring.strip(), re.IGNORECASE))

def is_auto_generated(code):
    # Check for auto-generated indicators
    return "# auto-generated" in code.lower() or "@generated" in code.lower()

def count_non_empty_lines(code):
    return len([line for line in code.splitlines() if line.strip()])

# === Filtering Logic ===

filtered = []
with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        try:
            item = json.loads(line)
            code = item.get("code", "")
            docstring = item.get("docstring", "")
            file_path = item.get("file", "")

            # Apply filters
            if not code or not docstring:
                continue
            if is_trivial_docstring(docstring):
                continue
            if is_auto_generated(code):
                continue
            if not (3 <= count_non_empty_lines(code) <= 30):
                continue
            if not (1 <= len(docstring.splitlines()) <= 3):
                continue

            # Normalize 'file' field to start from repo root
            if file_path.startswith(repo_root):
                item["file"] = file_path[len(repo_root):]

            filtered.append(item)
        except json.JSONDecodeError:
            continue  # skip malformed lines

# === Save Filtered Output ===
with open(output_file, "w", encoding="utf-8") as out_f:
    for entry in filtered:
        out_f.write(json.dumps(entry) + "\n")

print(f"Filtered {len(filtered)} pairs saved to: {output_file}")