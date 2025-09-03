import pandas as pd
from sentence_transformers import SentenceTransformer, models, InputExample, losses, util
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys, os

# ====== Configuration ======
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CSV_INPUT_PATH = '/home/user/projects/prompt-project/SIDE_p/scripts_and_data/training_SIDE/evaluation_and_statistical_tests/NEW-generation_from_3_models/CS-benchmark-Python-nohw.csv'
CSV_OUTPUT_PATH = '/home/user/projects/prompt-project/SIDE_p/scripts_and_data/training_SIDE/evaluation_and_statistical_tests/NEW-generation_from_3_models/CS-benchmark-Python-nohw.csv'  
#CHECKPOINT_FOLDER = '/home/user/projects/prompt-project/SIDE_p/scripts_and_data/training_SIDE/models/triplet-loss/no_hard_negatives/checkpoint-118050'
CHECKPOINT_FOLDER = '/scratch/user/projects/SIDE-P-trained-models/models/triplet-loss/no_hard_negatives/checkpoint-118050'  # moved models to scratch due to space limitation
#CHECKPOINT_FOLDER ="microsoft/mpnet-base" 

# ====== Mean Pooling ======
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
           torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# ====== Load Model ======
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_FOLDER)
model = AutoModel.from_pretrained(CHECKPOINT_FOLDER).to(DEVICE)

# ====== Load CSV ======
df = pd.read_csv(CSV_INPUT_PATH)
side_scores = []

# ====== Compute SIDE scores ======
print("Computing SIDE scores for each instance...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    #code = str(row["codeFunctions"])
    #summary = str(row["codeComment"])
    code = str(row["target"])
    summary = str(row["summary_postprocessed"])
    pair = [summary,code]

    encoded_input = tokenizer(pair, padding=True, truncation=True, return_tensors='pt').to(DEVICE)

    # Compute embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform mean pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    # Compute cosine similarity
    sim_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    # Append the score to the list
    side_scores.append(sim_score)
    '''print("*********************************************")
    print("Index: ", idx)
    print("Code: ", code)
    print("Generated Summary: ", summary)
    print("SIDE Score: ", sim_score)
    print("*********************************************\n")'''

# ====== Write SIDE scores to CSV ======
df["SIDE_score"] = side_scores
df.to_csv(CSV_OUTPUT_PATH, index=False)

# ====== Report Overall Score ======
average_score = sum(side_scores) / len(side_scores)
print(f"\n Average SIDE score across all instances: {average_score:.4f}")
print(f" Updated CSV saved to: {CSV_OUTPUT_PATH}")