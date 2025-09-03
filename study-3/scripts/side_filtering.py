import csv
import sys
import warnings
#from noise_detection import *
from bs4 import BeautifulSoup
import json
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import os
import re
import json
import torch.nn.functional as F
from sentence_transformers import util

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
checkPointFolder = "/scratch/saima/projects/SIDE-P-trained-models/models/triplet-loss/no_hard_negatives/checkpoint-118050"
tokenizer = AutoTokenizer.from_pretrained(checkPointFolder)
model = AutoModel.from_pretrained(checkPointFolder).to(DEVICE)


def update_BlockComment(raw_code):
    p = re.compile('^(\s+//)|(//)')
    new_list = []
    for row in raw_code.split('\n'):
        if not p.search(row):
            new_list.append(row)
    return '\n'.join(new_list)


def update_ContentTamper(comment):
    return BeautifulSoup(comment, "html.parser").get_text()


def mean_pooling(output, attention_mask):
    token_embeddings = output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def side_score(pair):
    encoded_input = tokenizer([pair['right_code'], pair['right_comment_format']], padding=True,
                              truncation=True, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    sim = util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[1]).item()
    return sim


def add_columns(pairs):
    modified_pairs = []
    for pair in tqdm(pairs.to_dict(orient='records'), total=len(pairs)):
        # pair['right_code'] = update_BlockComment(pair['raw_code'])
        # pair['right_comment_format'] = update_ContentTamper(pair['raw_comment'])
        #pair['right_code'] = update_BlockComment(' '.join(pair['code']))
        pair['right_code'] = update_BlockComment(' '.join(pair['code_tokens']))
        pair['right_comment_format'] = update_ContentTamper(pair['docstring'])
        #pair['right_comment_format'] = getFirstSentence(pair['right_comment_format'])
        pair['SIDE_score'] = side_score(pair)

        # del pair['code']
        # del pair['comment']

        # Clean up only if needed; skip 'comment' since it doesn't exist
        #pair.pop('code', None)  # safe delete
        modified_pairs.append(pair)

    return pd.DataFrame(modified_pairs)


def handle_spits(pairs, split, threshold):
    filtered_pairs = []
    excluded = []
    for pair in tqdm(pairs.to_dict(orient='records'), total=len(pairs), desc=f"Processing {split} pairs"):
        if pair['SIDE_score'] >= threshold:
            filtered_pairs.append(pair)
        else:
            excluded.append(pair)

    return filtered_pairs, excluded

# Function to write DataFrame to JSONL format
def write_jsonl(df, path):
        with open(path, 'w', encoding='utf-8') as f:
            for row in df.to_dict(orient='records'):
                f.write(json.dumps(row) + '\n')

def write_res(filtered, excluded, split, threshold, dataset_name):
    df_filtered = pd.DataFrame(filtered)
    #df_filtered.to_csv(f'output/{dataset_name}/threshold_{threshold}/{split}_filtered_more_than_{threshold}.csv', index=False, encoding='utf-8', escapechar='\\', quoting=csv.QUOTE_NONNUMERIC)
    
    df_excluded = pd.DataFrame(excluded)
    #df_excluded.to_csv(f'output/{dataset_name}/threshold_{threshold}/{split}_excluded_w_less_than_{threshold}.csv', index=False, encoding='utf-8', escapechar='\\', quoting=csv.QUOTE_NONNUMERIC)
    write_jsonl(df_filtered, os.path.join(threshold_dir, f"{split}_filtered_more_than_{threshold}.jsonl"))
    write_jsonl(df_excluded, os.path.join(threshold_dir, f"{split}_excluded_below_{threshold}.jsonl"))

def clean_and_load_json(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                cleaned_line = json.loads(line)
                data.append(cleaned_line)
            except json.JSONDecodeError:
                print("Error decoding JSON line, skipping: ", line)
                continue
    return pd.DataFrame(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process SIDE score threshold.')
    parser.add_argument('--dataset-name', type=str, default='funcom_python',
                        help='Name of the dataset')
    parser.add_argument('--output-dir', type=str, default='output',
                    help='Root folder to store filtered results')
    args = parser.parse_args()

    # train_path = f'Are_we_building_on_the_rock_datasets/clean/{args.dataset_name}/train/{args.dataset_name}.train'
    # valid_path = f'Are_we_building_on_the_rock_datasets/clean/{args.dataset_name}/valid/{args.dataset_name}.valid'
    # test_path = f'Are_we_building_on_the_rock_datasets/clean/{args.dataset_name}/test/{args.dataset_name}.test'
    
    train_path = '/home/saima/projects/prompt-project/codet5+/dataset/funcom-python-final/CAT-cleaned-no-SIDE/Base+AST/funcom_python_train_CAT-cleaned.jsonl'
    valid_path = '/home/saima/projects/prompt-project/codet5+/dataset/funcom-python-final/CAT-cleaned-no-SIDE/Base+AST/funcom_python_valid_CAT-cleaned.jsonl'
    test_path = '/home/saima/projects/prompt-project/codet5+/dataset/funcom-python-final/CAT-cleaned-no-SIDE/Base+AST/funcom_python_test_CAT-cleaned.jsonl'

    def load_jsonl(path):
        with open(path, "r") as f:
            return pd.DataFrame([json.loads(line) for line in f])

    df_train = load_jsonl(train_path)
    df_valid = load_jsonl(valid_path)
    df_test = load_jsonl(test_path)

    pairs_train = add_columns(df_train)
    pairs_valid = add_columns(df_valid)
    pairs_test = add_columns(df_test)

    # pairs_train = add_columns(clean_and_load_json(train_path))
    # pairs_valid = add_columns(clean_and_load_json(valid_path))
    # pairs_test = add_columns(clean_and_load_json(test_path))

    # os.makedirs(f'output/{args.dataset_name}/', exist_ok=True)
    # pairs_train.to_csv(f'output/{args.dataset_name}/train.csv', index=False, encoding='utf-8', escapechar='\\', quoting=csv.QUOTE_NONNUMERIC)
    # pairs_valid.to_csv(f'output/{args.dataset_name}/validation.csv', index=False, encoding='utf-8', escapechar='\\', quoting=csv.QUOTE_NONNUMERIC)
    # pairs_test.to_csv(f'output/{args.dataset_name}/test.csv', index=False, encoding='utf-8', escapechar='\\', quoting=csv.QUOTE_NONNUMERIC)


    #os.makedirs(f'output/{args.dataset_name}/', exist_ok=True)
    base_path = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(base_path, exist_ok=True)
    write_jsonl(pairs_train, os.path.join(base_path, 'train.jsonl'))
    write_jsonl(pairs_train, os.path.join(base_path, 'valid.jsonl'))
    write_jsonl(pairs_train, os.path.join(base_path, 'test.jsonl'))
    # write_jsonl(pairs_train, f'output/{args.dataset_name}/train.jsonl')
    # write_jsonl(pairs_valid, f'output/{args.dataset_name}/validation.jsonl')
    # write_jsonl(pairs_test, f'output/{args.dataset_name}/test.jsonl')

    for threshold in [0.5,0.6,0.7,0.8,0.9]:
        format_threshold = str(threshold).replace('.', '_')
        print(f'====================== Writing {threshold} ======================')
        #os.makedirs(f'output/{args.dataset_name}/threshold_{format_threshold}', exist_ok=True)
        threshold_dir = os.path.join(base_path, f"threshold_{format_threshold}")
        os.makedirs(threshold_dir, exist_ok=True)
        
        training_filtered, training_excluded = handle_spits(pairs_train, "training", threshold)
        validation_filtered, validation_excluded = handle_spits(pairs_valid, "validation", threshold)
        test_filtered, test_excluded = handle_spits(pairs_test, "test", threshold)
    
        # write_res(training_filtered, training_excluded, "train", format_threshold, args.dataset_name)
        # write_res(validation_filtered, validation_excluded, "val", format_threshold, args.dataset_name)
        # write_res(test_filtered, test_excluded, "test", format_threshold, args.dataset_name)
        write_res(training_filtered, training_excluded, "train", format_threshold, threshold_dir)
        write_res(validation_filtered, validation_excluded, "val", format_threshold, threshold_dir)
        write_res(test_filtered, test_excluded, "test", format_threshold, threshold_dir)
