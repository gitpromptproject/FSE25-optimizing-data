# ********** This script extract all the <code,comment> pairs from the repos in the root_dir

import os
import ast
import json

# === CONFIGURATION ===
root_dir = "/home/user/projects/prompt-project/SIDE_p/python-benchmark-repos"
output_file = "./data_files/new_python_code_comment_pairs.jsonl"  # saves in current directory

# === Helper Function ===
def extract_code_comment_pairs(file_path):
    pairs = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    function_code = ast.get_source_segment(source, node)
                    if function_code:
                        pairs.append({
                            "file": file_path,
                            "function_name": node.name,
                            "code": function_code.strip(),
                            "docstring": docstring.strip()
                        })
    except Exception as e:
        print(f"Skipped {file_path}: {e}")
    return pairs

# === Main Walker ===
all_pairs = []
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".py"):
            filepath = os.path.join(dirpath, filename)
            all_pairs.extend(extract_code_comment_pairs(filepath))

# === Save Output ===
with open(output_file, "w", encoding="utf-8") as out_f:
    for item in all_pairs:
        out_f.write(json.dumps(item) + "\n")

print(f" Extracted {len(all_pairs)} code-comment pairs")
print(f" Saved to: {output_file}")
