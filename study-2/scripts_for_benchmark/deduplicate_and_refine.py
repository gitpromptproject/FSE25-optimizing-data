"""
---------- Deduplication and Code Refinement-----------
After running "apply_cat_cleaning.py" script to use CAT tool to filter clean code,

This script removes duplicate or near-duplicate Python methods and standardizes code fields.

Steps:
1. **Deduplication**:
   - Use hash-based (e.g., SHA256) or semantic similarity (e.g., MinHash, cosine) to filter out similar methods.

2. **Code Field Cleanup**:
   - Rename original 'code' field to 'raw_code', 'docstring' field to 'raw_docstring'
   - Extract implementation-only code by removing the docstring from clean_code
   - Store the only-code version in a new 'input_code' field

Example:
{
  "raw_code": "def foo():\n    \"\"\"docstring\"\"\"\n    do_something()",
  "code":     "def foo():\n    do_something()"
}
"""

import json
import hashlib
import ast
import re
import textwrap

# === Configuration ===
input_file = "/home/user/projects/prompt-project/SIDE_p/scripts_and_data/data_files/extension/extended_cleaned_code_comment_pairs_CAT-4000.jsonl"
output_file = "/home/user/projects/prompt-project/SIDE_p/scripts_and_data/data_files/extension/extended_cleaned_deduplicated_code_comment_pairs-4000.jsonl"

# === Helper Functions ===

def strip_docstring_from_function(code: str) -> str:
    """
    Removes any triple-quoted strings (docstrings) from the code.
    Handles """, r""", and ''' variations.
    """
    # Pattern to match triple-quoted strings with optional r prefix
    # This handles:
    # - """...""" 
    # - r"""..."""
    # - '''...'''
    # - r'''...'''
    # Both single-line and multi-line
    
    patterns = [
        # Triple double quotes (multi-line and single-line)
        r'r?"""[\s\S]*?"""',
        # Triple single quotes (multi-line and single-line)  
        r"r?'''[\s\S]*?'''"
    ]
    
    result = code
    for pattern in patterns:
        result = re.sub(pattern, '', result, flags=re.MULTILINE | re.DOTALL)
    
    return result

def hash_code(code):
    """Generate a hash for the code string."""
    return hashlib.sha256(code.strip().encode("utf-8")).hexdigest()

def clean_docstring(docstring):
    """Remove GitHub/website links and URLs from docstring."""
    return re.sub(r'https?://\S+|www\.\S+|\S+\.(com|org|io|net|edu|gov)\S*', '', docstring).strip()

# === Main Processing ===

seen_hashes = set()
deduplicated_entries = []

with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        try:
            entry = json.loads(line)
            # Extract original values
            raw_code = entry.get("code", "")
            raw_docstring = entry.get("docstring", "")
            full_code = entry.get("clean_code", "")
            docstring = entry.get("clean_docstring", "")

            if not full_code.strip() or not docstring.strip():
                continue

            # Clean docstring
            cleaned_docstring = clean_docstring(docstring)

            # Create final structured entry
            entry["raw_code"] = raw_code
            entry["raw_docstring"] = raw_docstring
            entry["clean_code"] = full_code
            entry["clean_docstring"] = docstring
            entry["summary"] = cleaned_docstring

            # Remove docstring from code
            final_code = strip_docstring_from_function(full_code)
            entry["input_code"] = final_code

            # Remove old fields
            entry.pop("code", None)
            entry.pop("docstring", None)

            # Deduplicate using hash of final_code
            code_hash = hash_code(final_code)
            if code_hash not in seen_hashes:
                seen_hashes.add(code_hash)
                deduplicated_entries.append(entry)

        except json.JSONDecodeError:
            continue  # skip malformed lines

# === Save Output ===
with open(output_file, "w", encoding="utf-8") as out_f:
    for item in deduplicated_entries:
        out_f.write(json.dumps(item) + "\n")

print(f"✅ Deduplicated and refined {len(deduplicated_entries)} code-comment pairs saved to: {output_file}")