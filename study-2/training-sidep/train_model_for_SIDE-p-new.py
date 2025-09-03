import os
import json
import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models, util, LoggingHandler

# ========== Setup ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ========== Config ==========
TRAIN_FILE = "/home/user/projects/prompt-project/SIDE_p/scripts/training_SIDE/side_finetune_codexglue.json"
VAL_FILE = "/home/user/projects/prompt-project/SIDE_p/scripts/training_SIDE/side_finetune_codexglue_valid.json"
OUTPUT_PATH = "models/mpnet_triplet_no_hardneg_v2-test"
BATCH_SIZE = 16
EPOCHS = 10
MAX_SEQ_LENGTH = 512
PATIENCE = 5
CHECKPOINT_STEPS = 5000
BEST_SCORE = float('-inf')
NO_IMPROVEMENT = 0

# ========== Logging Setup ==========
log_file = os.path.join(OUTPUT_PATH, "training.log")
# Create file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)

# Set up basic logging (console + file)
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler(), file_handler]
)

# ========== Load Data ==========
def load_data(path):
    with open(path) as f:
        data = json.load(f)
    examples = [InputExample(texts=[item["query"], item["pos"], item["neg"]]) for item in data]
    return examples

train_examples = load_data(TRAIN_FILE)
val_examples = load_data(VAL_FILE)
print(len(train_examples), "train examples loaded")
print(len(val_examples), "validation examples loaded")

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

# ========== Model ==========
word_embedding_model = models.Transformer("microsoft/mpnet-base", max_seq_length=MAX_SEQ_LENGTH)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_loss = losses.TripletLoss(model=model)

# ========== Warmup ==========
total_steps = len(train_dataloader) * EPOCHS
warmup_steps = int(total_steps * 0.1)

# ========== Evaluation ==========
def evaluate_val_score(model, val_data):
    scores = []
    for example in tqdm(val_data, desc="Evaluating"):
        query_emb = model.encode(example.texts[0], convert_to_tensor=True)
        pos_emb = model.encode(example.texts[1], convert_to_tensor=True)
        neg_emb = model.encode(example.texts[2], convert_to_tensor=True)

        cs_pos = util.cos_sim(query_emb, pos_emb).item()
        cs_neg = util.cos_sim(query_emb, neg_emb).item()
        scores.append(cs_pos - cs_neg)

    return sum(scores) / len(scores)

# ========== Training Loop with Manual Early Stopping ==========
for epoch in range(EPOCHS):
    logging.info(f"\n======== Epoch {epoch + 1} / {EPOCHS} ========")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=warmup_steps,
        output_path=None,  # Final best model saved manually below
        checkpoint_path=OUTPUT_PATH,
        checkpoint_save_steps=CHECKPOINT_STEPS,
        show_progress_bar=True,
        #use_amp=True
    )

    val_score = evaluate_val_score(model, val_examples)
    logging.info(f"Validation Score after epoch {epoch+1}: {val_score:.4f}")

    if val_score > BEST_SCORE:
        BEST_SCORE = val_score
        NO_IMPROVEMENT = 0
        model.save(OUTPUT_PATH)
        logging.info(f"New best model saved with score: {BEST_SCORE:.4f}")
    else:
        NO_IMPROVEMENT += 1
        logging.info(f"No improvement. Patience counter: {NO_IMPROVEMENT}/{PATIENCE}")

    if NO_IMPROVEMENT >= PATIENCE:
        logging.info("Early stopping triggered.")
        break

# ========== Final Output ==========
logging.info(f"\nBest validation score: {BEST_SCORE:.4f}")