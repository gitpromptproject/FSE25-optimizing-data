import json
import csv

def remove_functions_from_jsonl(jsonl_file, csv_file, output_file):
   # Read CSV file to get functions to remove
   functions_to_remove = set()
   code_functions_to_remove = set()
   original_comments_to_remove = set()
   
   with open(csv_file, 'r') as f:
       reader = csv.DictReader(f)
       for row in reader:
           if 'function_name' in row and row['function_name']:
               functions_to_remove.add(row['function_name'])
           if 'codeFunctions' in row and row['codeFunctions']:
               code_functions_to_remove.add(row['codeFunctions'])
           if 'originalComment' in row and row['originalComment']:
               original_comments_to_remove.add(row['originalComment'])
   
   # Process JSONL file
   removed_count = 0
   total_count = 0
   
   with open(jsonl_file, 'r') as infile, open(output_file, 'w') as outfile:
       for line in infile:
           total_count += 1
           data = json.loads(line.strip())
           
           # Check if should be removed based on any of the three criteria
           should_remove = (
               data.get('function_name') in functions_to_remove or
               data.get('clean_code') in code_functions_to_remove or
               data.get('clean_docstring') in original_comments_to_remove
           )
           
           if not should_remove:
               outfile.write(json.dumps(data) + '\n')
           else:
               removed_count += 1
   
   print(f"Total entries processed: {total_count}")
   print(f"Entries removed: {removed_count}")
   print(f"Entries kept: {total_count - removed_count}")

# Usage
remove_functions_from_jsonl('extended_cleaned_code_comment_pairs_CAT-4000.jsonl', '/home/user/projects/prompt-project/SIDE_p/scripts_and_data/training_SIDE/evaluation_and_statistical_tests/benchmark-gpt-evaluation/gpt_annotation-0.csv', 'filtered_after500_method.jsonl')