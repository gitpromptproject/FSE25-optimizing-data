import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 


# === Helper ===
def clean_tokens(text):
    return text.replace("</s>", "").replace("<pad>", "").strip()

# === Dataset ===
class SignaturePromptDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, args):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_source_length = args.encoder_block_size
        self.max_target_length = args.decoder_block_size

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                #input_text = ' '.join(data['signature_prompt'])  # already tokenized
                #input_text = ' '.join(data['shortened_code'])  # already tokenized
                #input_text = ' '.join(data['code_tokens'])  # already tokenized
                input_text = ' '.join(data['ast_prompt'])  # already tokenized
                target_text = ' '.join(data['docstring_tokens'])  # already tokenized
                self.samples.append((input_text, target_text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_text, target_text = self.samples[idx]
        source = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        target = self.tokenizer.encode_plus(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return (
            source['input_ids'].squeeze(0),
            source['attention_mask'].squeeze(0),
            target['input_ids'].squeeze(0)
        )

# === Inference ===
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    dataset = SignaturePromptDataset(args.input_file, tokenizer, args)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.eval_batch_size)

    raw_predictions = []
    groundtruth_sentence = []
    accuracy = []

    for batch in tqdm(dataloader, desc="Running Inference"):
        input_ids, attention_mask, target_ids = [x.to(device) for x in batch]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=args.num_beams,
                num_return_sequences=1,
                max_length=args.decoder_block_size,
                do_sample=False
            )

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        decoded_targets = tokenizer.batch_decode(target_ids, skip_special_tokens=False)

        for pred, gt in zip(decoded_preds, decoded_targets):
            pred_clean = clean_tokens(pred)
            gt_clean = clean_tokens(gt)

            raw_predictions.append(pred_clean)
            groundtruth_sentence.append(gt_clean)
            accuracy.append(int(''.join(pred_clean.lower().split()) == ''.join(gt_clean.lower().split())))

    # === Save ===
    prompt_types = args.prompt_types
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame({
        "target": groundtruth_sentence,
        "raw_predictions": raw_predictions,
        "correctly_predicted": accuracy
    })
    df.to_csv(os.path.join(args.output_dir, f"{prompt_types}_benchmark_inference_results-test.csv"), index=False)

    with open(os.path.join(args.output_dir, f"{prompt_types}_benchmark_inference_results-test.jsonl"), "w") as f:
        for tgt, pred in zip(groundtruth_sentence, raw_predictions):
            f.write(json.dumps({"target": tgt, "raw_predictions": pred}) + "\n")

    #print(f"Accuracy: {round(sum(accuracy) / len(accuracy), 4)}")
    print(f"Saved results to {args.output_dir}")


# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="Salesforce/codet5p-220m")
    parser.add_argument("--output_dir", type=str, default="./inference_output")
    parser.add_argument("--encoder_block_size", type=int, default=512)
    parser.add_argument("--decoder_block_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt_types", type=str, required=True)
    args = parser.parse_args()

    run_inference(args)