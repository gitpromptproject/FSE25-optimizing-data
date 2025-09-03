import json
import tokenize
import io
import random
import csv

# === Configuration ===
input_file = "./data_files/new_deduplicated_code_comment_pairs.jsonl"
jsonl_output_file = "./data_files/new_latest_python_code_comment_pairs_benchmark.jsonl"
csv_output_file = "./data_files/new_latest_python_code_comment_pairs_benchmark.csv"
sample_size = 4000  # number of pairs to sample
min_tokens = 20  

# === Helper Functions ===

def tokenize_python_code(code_str):
    """Tokenize Python code using Python's built-in tokenizer."""
    try:
        tokens = [tok.string for tok in tokenize.generate_tokens(io.StringIO(code_str).readline)]
        return [t for t in tokens if t.strip()]
    except Exception:
        return []

def simple_tokenize_docstring(docstring):
    """Simple whitespace tokenizer for docstrings."""
    return docstring.strip().split()

# === Load and Process ===

entries = []
removed_count = 0
with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        try:
            entry = json.loads(line)
            code_tokens = tokenize_python_code(entry.get("input_code", ""))
            docstring_tokens = simple_tokenize_docstring(entry.get("summary", ""))
            #raw_code = entry.get("code", "")

            if len(code_tokens) > min_tokens and len(docstring_tokens) > 3:
                entry["code_tokens"] = code_tokens
                entry["docstring_tokens"] = docstring_tokens
                #entry["raw_code"] = raw_code
                entries.append(entry)
            else:
                removed_count += 1
        except json.JSONDecodeError:
            continue
            
print(f"Filtered out {removed_count} entries with < {min_tokens} code tokens")
print(f"Remaining entries: {len(entries)}")

# === Random Sampling ===
random.seed(42)
sampled = random.sample(entries, min(sample_size, len(entries)))

# === Save JSONL ===
with open(jsonl_output_file, "w", encoding="utf-8") as jsonl_out:
    for item in sampled:
        jsonl_out.write(json.dumps(item) + "\n")

# === Save CSV ===
csv_fields = [
    "file",
    "function_name",
    "raw_code",
    "clean_code",
    "raw_docstring",
    "clean_docstring",
    "input_code",
    "summary",
    "code_tokens",
    "docstring_tokens",
    
]

with open(csv_output_file, "w", encoding="utf-8", newline="") as csv_out:
    writer = csv.DictWriter(csv_out, fieldnames=csv_fields)
    writer.writeheader()
    for item in sampled:
        writer.writerow({
            "file": item.get("file", ""),
            "function_name": item.get("function_name", ""),
            "raw_code": item.get("raw_code", ""),
            "clean_code": item.get("clean_code", ""),
            "raw_docstring": item.get("raw_docstring", ""),
            "clean_docstring": item.get("clean_docstring", ""),
            "input_code": item.get("input_code", ""),
            "summary": item.get("summary", ""),
            "code_tokens": " ".join(item.get("code_tokens", [])),
            "docstring_tokens": " ".join(item.get("docstring_tokens", [])),
        })

print(f" Sampled {len(sampled)} code-comment pairs")
print(f" JSONL saved to: {jsonl_output_file}")
print(f" CSV saved to: {csv_output_file}")