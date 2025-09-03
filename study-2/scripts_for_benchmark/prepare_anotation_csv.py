import json
import csv

# === Configuration ===
input_file = "./data_files/new_latest_python_code_comment_pairs_benchmark.jsonl"
output_csv = "./data_files/new_human_anotation_python_code_summaries.csv"

# === Load and process entries ===
entries = []
with open(input_file, "r", encoding="utf-8") as infile:
    for idx, line in enumerate(infile):
        try:
            item = json.loads(line)
            entries.append({
                "id": idx + 1,
                "file": item.get("file", ""),
                "function_name": item.get("function_name", ""),
                "raw_code": item.get("raw_code", ""),
                "clean_code": item.get("clean_code", ""),
                "raw_docstring": item.get("raw_docstring", ""),
                "clean_docstring": item.get("clean_docstring", ""),
                "input_code": item.get("input_code", ""),
                "summary": item.get("summary", ""),
                "conciseness_score": "",
                "fluency_score": "",
                "content_adequacy_score": ""
            })
        except json.JSONDecodeError:
            continue

# === Define CSV fields ===
fieldnames = [
    "id", "file", "function_name", "raw_code", "clean_code",
    "raw_docstring", "clean_docstring", "input_code", "summary",
    "conciseness_score", "fluency_score", "content_adequacy_score"
]

# === Write to CSV ===
with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in entries:
        writer.writerow(row)

print(f" Human evaluation CSV generated with {len(entries)} entries.")
print(f" File saved to: {output_csv}")