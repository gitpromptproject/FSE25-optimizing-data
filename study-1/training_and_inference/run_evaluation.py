import json
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import evaluate

# === Load metrics ===
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
chrf = evaluate.load("chrf")
#lemmatizer = WordNetLemmatizer()

# === Input/output paths
input_csv_file = "/scratch/user/projects/SIDE-funcom-java-models/AST/evaluation-test/checkpoint-101283.csv"
#input_file = "/scratch/user/projects/raw-funcom-java-models/BASE-test/checkpoint-step-148056/evaluation-2/checkpoint--1.csv"
output_file = "/scratch/user/projects/SIDE-funcom-java-models/AST/evaluation-test/checkpoint-101283.txt"

# === Helper function
'''def normalize_and_lemmatize(text):
    tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(tok) for tok in tokens]'''

# === Load predictions and references from CSV
predictions, references = [], []
# em_scores = []

df = pd.read_csv(input_csv_file)
for _, row in tqdm(df.iterrows(), total=len(df), desc="Reading predictions from CSV"):
    prediction = str(row.get("raw_predictions", "")).strip()
    reference = str(row.get("target", "")).strip()

    if not prediction or not reference:
        continue

    predictions.append(prediction)
    references.append(reference)

    # pred_tokens = normalize_and_lemmatize(prediction)
    # ref_tokens = normalize_and_lemmatize(reference)
    # em_scores.append(1.0 if pred_tokens == ref_tokens else 0.0)

'''# === Load predictions and references from JSONL
predictions, references = [], []
#em_scores = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading predictions"):
        entry = json.loads(line.strip())
        prediction = entry.get("raw_predictions", "").strip()
        reference = entry.get("target", "").strip()

        if not prediction or not reference:
            continue

        predictions.append(prediction)
        references.append(reference)

        #pred_tokens = normalize_and_lemmatize(prediction)
        #ref_tokens = normalize_and_lemmatize(reference)
        #em_scores.append(1.0 if pred_tokens == ref_tokens else 0.0)'''

# === Compute metrics
bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in references])
rouge_result = rouge.compute(predictions=predictions, references=references)
meteor_result = meteor.compute(predictions=predictions, references=references)
chrf_result = chrf.compute(predictions=predictions, references=references)

# === Save results
with open(output_file, "w") as f_out:
    f_out.write("Evaluation Metrics:\n")
    f_out.write(f"BLEU: {bleu_result['bleu']:.4f}\n")
    f_out.write(f"ROUGE-L: {rouge_result['rougeL']:.4f}\n")
    f_out.write(f"METEOR: {meteor_result['meteor']:.4f}\n")
    f_out.write(f"ChrF: {chrf_result['score'] / 100:.4f}\n")
    #f_out.write(f"Exact Match (Lemmatized): {sum(em_scores)/len(em_scores):.4f}\n")

print(f" Evaluation complete.\n Results saved to: {output_file}")