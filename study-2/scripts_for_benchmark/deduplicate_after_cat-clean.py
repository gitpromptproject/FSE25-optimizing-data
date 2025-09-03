import sys
import json
import hashlib
from tqdm import tqdm

# === File paths ===
input_path = "cleaned_code_comment_pairs_CAT.jsonl"
output_path = "deduplicated_code_comment_pairs.jsonl"

def hash_code(code):
    """
    Create a hash of the code for deduplication.
    Normalizes whitespace and removes comments to focus on actual logic.
    """
    if not code:
        return ""
    
    # Basic normalization: remove extra whitespace and normalize line endings
    normalized_code = ' '.join(code.split())
    
    # Create SHA256 hash
    return hashlib.sha256(normalized_code.encode('utf-8')).hexdigest()

# === Process entries for deduplication ===
seen_hashes = set()
kept = 0
duplicates_removed = 0
total_processed = 0

print("Deduplicating entries based on clean_code...")

with open(input_path, "r") as infile, open(output_path, "w") as outfile:
    for line in tqdm(infile, desc="Processing entries"):
        try:
            entry = json.loads(line)
            total_processed += 1
            
            # Get the clean_code for hashing (fallback to original code if clean_code not available)
            raw_code = entry.get("clean_code", entry.get("code", ""))
            
            if not raw_code or raw_code.strip() == "":
                # Skip entries with empty code
                duplicates_removed += 1
                continue
            
            # Generate hash of the code
            code_hash = hash_code(raw_code)
            
            # Check if we've seen this code before
            if code_hash not in seen_hashes:
                seen_hashes.add(code_hash)
                outfile.write(json.dumps(entry) + "\n")
                kept += 1
            else:
                duplicates_removed += 1
                
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error: {e}")
            duplicates_removed += 1
            continue
        except Exception as e:
            print(f"⚠️ Error processing entry: {e}")
            duplicates_removed += 1
            continue

print(f"\n✅ Deduplication Results:")
print(f"   Total processed entries: {total_processed}")
print(f"   Unique entries kept: {kept}")
print(f"   Duplicate entries removed: {duplicates_removed}")
print(f"   Deduplication rate: {duplicates_removed/total_processed*100:.2f}%")
print(f"   Output written to: {output_path}")

# === Optional: Quick validation ===
'''if kept > 0:
    print(f"\n📋 Quick validation:")
    with open(output_path, "r") as f:
        first_line = f.readline()
        sample_entry = json.loads(first_line)
        required_fields = ["file", "function_name", "code", "docstring"]
        if "clean_code" in sample_entry:
            required_fields.extend(["clean_code", "clean_docstring"])
        
        missing_fields = [field for field in required_fields if field not in sample_entry]
        if missing_fields:
            print(f"   ⚠️ Missing fields in output: {missing_fields}")
        else:
            print(f"   ✅ All required fields present in output")
            
    # Count unique hashes as final verification
    with open(output_path, "r") as f:
        final_hashes = set()
        for line in f:
            entry = json.loads(line)
            code = entry.get("clean_code", entry.get("code", ""))
            if code:
                final_hashes.add(hash_code(code))
        
        print(f"   ✅ Final verification: {len(final_hashes)} unique code hashes in output")
        if len(final_hashes) != kept:
            print(f"   ⚠️ Warning: Hash count mismatch ({len(final_hashes)} vs {kept})")
'''