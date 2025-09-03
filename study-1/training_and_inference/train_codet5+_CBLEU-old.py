# Updated training script using codet5.py logic, adapted for your JSONL dataset structure

import os
import json
import logging
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class InputFeatures:
    def __init__(self, input_ids, label_ids, decoder_input_ids):
        self.input_ids = input_ids
        self.label = label_ids
        self.decoder_input_ids = decoder_input_ids

def convert_example_to_features(example, tokenizer, args):
    input_str = " ".join(example["shortened_code"]) if isinstance(example["shortened_code"], list) else example["shortened_code"]
    label_str = example["docstring"]

    input_ids = tokenizer.encode(input_str, truncation=True, max_length=args.encoder_block_size, padding='max_length', return_tensors='pt')
    decoder_input_ids = tokenizer.encode(label_str, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    label_ids = tokenizer.encode(label_str, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    return InputFeatures(input_ids, label_ids, decoder_input_ids)

class JSONLDataset(Dataset):
    def __init__(self, file_path, tokenizer, args):
        self.examples = []
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc=f"Loading {file_path}"):
                ex = json.loads(line)
                self.examples.append(convert_example_to_features(ex, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (self.examples[i].input_ids.squeeze(0),
                self.examples[i].input_ids.ne(0).squeeze(0),
                self.examples[i].label.squeeze(0),
                self.examples[i].decoder_input_ids.squeeze(0))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "").replace("<s>", "").replace("</s>", "").strip("\n").strip()
    return tokens

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.to(device)

    dataset_name = os.path.splitext(os.path.basename(args.train_file))[0]
    output_dir = os.path.join(os.path.dirname(args.train_file), f"{args.model_name}_{dataset_name}")
    args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = JSONLDataset(args.train_file, tokenizer, args)
    val_dataset = JSONLDataset(args.valid_file, tokenizer, args)

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size)
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args.eval_batch_size)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 5, num_training_steps=total_steps)

    writer = SummaryWriter(log_dir=args.tb_log_dir)
    best_val_loss = float('inf')

    logger.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
            input_ids, attention_mask, labels, decoder_input_ids = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels, decoder_input_ids = [x.to(device) for x in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved full model and tokenizer to {output_dir}")

    writer.close()

class Args:
    train_file = "/home/user/projects/prompt-project/codet5+/dataset/funcom/CBLEU/shortened_funcom_train_CBLEU_java.jsonl"
    valid_file = "/home/user/projects/prompt-project/codet5+/dataset/funcom/CBLEU/shortened_funcom_valid_CBLEU_java.jsonl"
    model_name_or_path = "Salesforce/codet5p-220m"
    tokenizer_name = "Salesforce/codet5p-220m"
    model_name = model_name_or_path.split("/")[-1]
    encoder_block_size = 256
    decoder_block_size = 64
    train_batch_size = 4
    eval_batch_size = 4
    learning_rate = 5e-5
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    epochs = 10
    seed = 42
    tb_log_dir = "tb_logs"

if __name__ == '__main__':
    args = Args()
    train(args)