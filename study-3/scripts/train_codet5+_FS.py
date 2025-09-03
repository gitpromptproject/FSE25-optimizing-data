# Updated training script using codet5.py logic, adapted for your JSONL dataset structure

import os
import json
import logging
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from fvcore.nn import FlopCountAnalysis
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 


logger = logging.getLogger(__name__)

class InputFeatures:
    def __init__(self, input_ids, label_ids, decoder_input_ids):
        self.input_ids = input_ids
        self.label = label_ids
        self.decoder_input_ids = decoder_input_ids

'''def convert_example_to_features(example, tokenizer, args):
    input_str = " ".join(example["signature_prompt"]) if isinstance(example["signature_prompt"], list) else example["signature_prompt"]
    label_str = example["docstring"]

    input_ids = tokenizer.encode(input_str, truncation=True, max_length=args.encoder_block_size, padding='max_length', return_tensors='pt')
    decoder_input_ids = tokenizer.encode(label_str, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    label_ids = tokenizer.encode(label_str, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    return InputFeatures(input_ids, label_ids, decoder_input_ids)
'''
def clean_unicode(text):
    # Replace invalid surrogates or non-UTF-8 with '?'
    return text.encode("utf-8", "replace").decode("utf-8")

def convert_example_to_features(example, tokenizer, args):
    # Handle both AST list and fallback string
    if isinstance(example.get("signature_prompt"), list):
        input_str = " ".join(example["signature_prompt"])
    else:
        input_str = example["signature_prompt"]

    input_str = clean_unicode(input_str)
    label_str = clean_unicode(example["docstring"])

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

def test(args, model, tokenizer, test_dataset, global_step):

    model.eval()
    device = args.device
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.eval_batch_size)

    raw_predictions = []
    groundtruth_sentence = []
    accuracy = []

    logger.info("***** Running Test *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")

    for batch in tqdm(test_loader, desc="Testing"):
        input_ids, attention_mask, labels, decoder_input_ids = [x.to(device) for x in batch]

        with torch.no_grad():
            beam_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                max_length=args.decoder_block_size,
                do_sample=False,
            )

        beam_outputs = beam_outputs.detach().cpu().tolist()
        decoder_input_ids = decoder_input_ids.detach().cpu().tolist()

        for idx, gt_ids in enumerate(decoder_input_ids):
            predictions = beam_outputs[idx * args.num_beams : (idx + 1) * args.num_beams]
            ground_truth = clean_tokens(tokenizer.decode(gt_ids, skip_special_tokens=False))
            correct_pred = False
            correct_prediction = ""

            for single_output in predictions:
                prediction = clean_tokens(tokenizer.decode(single_output, skip_special_tokens=False))
                if ''.join(prediction.lower().split()) == ''.join(ground_truth.lower().split()):
                    correct_prediction = prediction
                    correct_pred = True
                    break

            if correct_pred:
                raw_predictions.append(correct_prediction)
                accuracy.append(1)
            else:
                raw_pred = clean_tokens(tokenizer.decode(predictions[0], skip_special_tokens=False))
                raw_predictions.append(raw_pred)
                accuracy.append(0)

            groundtruth_sentence.append(ground_truth)

    # Accuracy computation and logging
    test_result = round(sum(accuracy) / len(accuracy), 4)
    logger.info("***** Test results *****")
    logger.info(f"Test Accuracy: {test_result}")

    # Save predictions in JSONL
    json_output_path = os.path.join(args.output_dir, f"golden-test-results@{global_step}.json")
    with open(json_output_path, "w") as f_json:
        for target, prediction in zip(groundtruth_sentence, raw_predictions):
            f_json.write(json.dumps({"target": target, "raw_predictions": prediction}) + "\n")

    # Save predictions and correctness in CSV
    csv_output_path = os.path.join(args.output_dir, f"golden-test-results@{global_step}.csv")
    df = pd.DataFrame({
        "target": groundtruth_sentence,
        "raw_predictions": raw_predictions,
        "correctly_predicted": accuracy
    })
    df.to_csv(csv_output_path, index=False)

    logger.info(f"Saved test predictions to:\n→ JSON: {json_output_path}\n→ CSV : {csv_output_path}")

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f"Using device: {device}")
    set_seed(args.seed)

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.to(device)

    dataset_name = os.path.splitext(os.path.basename(args.train_file))[0]
    #output_dir = os.path.join(os.path.dirname(args.train_file), f"{args.model_name}_{dataset_name}")
    output_dir = os.path.join(args.base_output_dir, f"{args.model_name}_{dataset_name}")
    final_model_dir = os.path.join(output_dir, "final_model")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)
    args.output_dir = output_dir

    # === Checkpointing setup ===
    best_checkpoint_path = os.path.join(output_dir, "checkpoint_best.pt")
    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    patience = 3
    delta = 0.01

    train_dataset = JSONLDataset(args.train_file, tokenizer, args)
    val_dataset = JSONLDataset(args.valid_file, tokenizer, args)
    test_dataset = JSONLDataset(args.test_file, tokenizer, args) if args.test_file else None

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size)
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args.eval_batch_size)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 5, num_training_steps=total_steps)

    # === Resume from best checkpoint if exists ===
    if os.path.exists(best_checkpoint_path):
        logger.info("Resuming from best checkpoint...")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        best_epoch = checkpoint["best_epoch"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]

    writer = SummaryWriter(log_dir=args.tb_log_dir)

    logger.info("Starting training...")
    total_training_flops = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        epoch_flops = 0  # New: FLOPs for this epoch
        step = 0
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
            input_ids, attention_mask, labels, decoder_input_ids = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # --- FLOPs measurement ---
            try:
                flops = FlopCountAnalysis(model, (input_ids, attention_mask, decoder_input_ids))
                flops.unsupported_ops_warnings(False)
                batch_flops = flops.total()
                total_training_flops += batch_flops
                epoch_flops += batch_flops
            except Exception as e:
                logger.warning(f"[FLOPs] Skipping FLOPs at step {step}: {e}")
            # --------------------------
            step += 1
            loss.backward()
            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # === FLOPs logging ===
        epoch_flops_giga = epoch_flops / 1e9
        print(f"[Epoch {epoch}] Total FLOPs: {epoch_flops_giga:.2f} GFLOPs")
        logger.info(f"[Epoch {epoch}] Total FLOPs: {epoch_flops_giga:.2f} GFLOPs")

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

        # === Save current checkpoint & clean previous
        current_ckpt = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "best_epoch": best_epoch,
        }, current_ckpt)

        # Delete previous checkpoint unless it's best
        if epoch > 0:
            prev_ckpt = os.path.join(output_dir, f"checkpoint_epoch_{epoch - 1}.pt")
            if os.path.exists(prev_ckpt) and prev_ckpt != best_checkpoint_path:
                os.remove(prev_ckpt)

        # === Check for improvement
        if best_val_loss - avg_val_loss > delta:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "best_epoch": best_epoch,
            }, best_checkpoint_path)
            model.save_pretrained(final_model_dir)
            tokenizer.save_pretrained(final_model_dir)
            logger.info(f"New best model saved to {final_model_dir}")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epochs.")

        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

        # === Optionally run test
        if args.test_on_eval_during_training and test_dataset:
            test(args, model, tokenizer, test_dataset, global_step=epoch)

    writer.close()
    logger.info(f"Best epoch: {best_epoch} | Best val_loss: {best_val_loss:.4f}")
    logger.info(f"Final best model restored from: {final_model_dir}")
    total_flops_giga = total_training_flops / 1e9
    logger.info(f"\n==== TOTAL TRAINING FORWARD FLOPs: {total_flops_giga:.2f} GFLOPs ====")

class Args:
    train_file = "/home/saima/projects/prompt-project/codet5+/dataset/funcom-python-final/CAT-cleaned-no-SIDE/FS/funcom_python_train_CAT-cleaned_FS.jsonl"
    valid_file = "/home/saima/projects/prompt-project/codet5+/dataset/funcom-python-final/CAT-cleaned-no-SIDE/FS/funcom_python_valid_CAT-cleaned_FS.jsonl"
    test_file = "/home/saima/projects/prompt-project/codet5+/dataset/funcom-python-final/CAT-cleaned-no-SIDE/FS/funcom_python_test_CAT-cleaned_FS.jsonl"
    model_name_or_path = "Salesforce/codet5p-220m"
    tokenizer_name = "Salesforce/codet5p-220m"
    base_output_dir = "/scratch/saima/projects/funcom-python-CATdata-models/no-SIDE/FS"
    model_name = model_name_or_path.split("/")[-1]
    encoder_block_size = 512
    decoder_block_size = 128
    train_batch_size = 16
    eval_batch_size = 8
    learning_rate = 2e-5
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    epochs = 15
    seed = 123456
    num_beams = 1
    tb_log_dir = "/scratch/saima/projects/funcom-python-CATdata-models/no-SIDE/FS/tb_logs"
    test_on_eval_during_training = True

if __name__ == '__main__':
    args = Args()

    # Set up logging to file and console
    log_path = os.path.join(args.base_output_dir, "training_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),       # Log to file
            logging.StreamHandler()              # Also print to console
        ]
    )

    train(args)