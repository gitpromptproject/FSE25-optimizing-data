import openai
import json
import csv
import time
from tqdm import tqdm

# === Configuration ===
#input_jsonl_path = "/home/user/projects/prompt-project/SIDE_p/scripts_and_data/training_SIDE/evaluation_and_statistical_tests/benchmark-gpt-evaluation/gpt_plus_human_annotation-500-old.csv"
input_csv_path = "/home/user/projects/prompt-project/SIDE_p/scripts_and_data/training_SIDE/evaluation_and_statistical_tests/benchmark-gpt-evaluation/gpt_generated_summaries_sample_50-old.csv"    
output_csv_path = "/home/user/projects/prompt-project/SIDE_p/scripts_and_data/training_SIDE/evaluation_and_statistical_tests/benchmark-gpt-evaluation/gpt_generated_summaries_sample_50-beginner.csv"
openai.api_key = ""  # Set your OpenAI API key here

# === Custom System Prompt ===
SYSTEM_PROMPT = (
    "Generate a single-line summary of the code surrounded by triple question marks (???). "
    "The summary should be semantic-focused and abstract, emphasizing the overall intent of the code. "
    "When composing the summary, naturalize identifiers (variable and function names) into meaningful keywords. "
    "Keep it highly concise—ideally around 15 tokens in length."
)


# === Function to query GPT-4o ===
def generate_summary(code_snippet):
    # Surround the code with triple question marks as per system prompt instruction
    formatted_input = f"???\n{code_snippet}\n???"
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": formatted_input}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during API call: {e}")
        return "API_ERROR"

# === Load data and run summarization ===
data = []
#with open(input_jsonl_path, 'r', encoding='utf-8') as infile:
with open(input_csv_path, 'r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in tqdm(reader, desc="Processing"):
        code = row.get("codeFunctions", "")
    #for line in tqdm(infile, desc="Processing"):
        #item = json.loads(line)
        #code = item.get("input_code", "")
        summary = generate_summary(code)
        # build new structure for output 
        '''output_row = {
            "id": item.get("id"),
            "function_name": item.get("function_name"),
            "codeFunctions": code,
            "originalComment": item.get("summary"),
            "codeComment": summary
        }'''
        output_row = {
            "id": row.get("id"),
            "function_name": row.get("function_name"),
            "codeFunctions": code,
            "originalComment": row.get("originalComment"),
            "codeComment-gpt3.5": summary
        }
        data.append(output_row)
        time.sleep(1.0)  # Optional: adjust based on your OpenAI rate limits

# === Write output to CSV ===
#fieldnames = list(data[0].keys())
fieldnames = ["id", "function_name", "codeFunctions", "originalComment", "codeComment"]
with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        writer.writerow(row)

print(f"✅ Output saved to {output_csv_path}")