from sentence_transformers import SentenceTransformer, models, InputExample, losses, util, LoggingHandler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import torch
import sys
import torch.nn.functional as F
import json
from sentence_transformers import evaluation
from tqdm import tqdm
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


'''logging.basicConfig(format='%(asctime)s - %(message)s',
					datefmt='%Y-%m-%d %H:%M:%S',
					level=logging.INFO,
					handlers=[LoggingHandler()])'''
OUTPUT_PATH ="models/triplet-loss/no_hard_negatives"
log_file =  os.path.join(OUTPUT_PATH + "training.log")
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_file),            # Logs to file
        LoggingHandler()                          # Still logs to console
    ]
)

def train(train_data, eval_data=None):

	n_train_examples = len(train_data)

	train_examples = []

	for i in range(n_train_examples):
		example = train_data[i]

		train_examples.append(InputExample(
			texts=[example['query'], example['pos'], example['neg']]))
		
		'''print("Query: ",example['query'])
		print("POS: ",example['pos'])
		print("NEG: ",example['neg'])
		sys.exit(0)'''

	train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

	# Step 1: use an existing language model
	word_embedding_model = models.Transformer('microsoft/mpnet-base')

	# Step 2: use a pool function over the token embeddings
	pooling_model = models.Pooling(
		word_embedding_model.get_word_embedding_dimension())

	# Join steps 1 and 2 using the modules argument
	model=SentenceTransformer(modules = [word_embedding_model, pooling_model])

	train_loss=losses.TripletLoss(model = model)

	num_epochs=15
	warmup_steps=int(len(train_dataloader) *
					   num_epochs * 0.1)  # 10% of train data

	model.fit(train_objectives = [(train_dataloader, train_loss)],
			  epochs=num_epochs,
			  checkpoint_save_steps=5000,
			  checkpoint_path=OUTPUT_PATH,
			  output_path=OUTPUT_PATH,
			  warmup_steps=warmup_steps,
			  show_progress_bar=True)


def main():
	with open('/home/user/projects/prompt-project/SIDE_p/scripts/training_SIDE/side_finetune_codexglue_train-new.json') as f:
		train_data = json.load(f)
	print("length of training data",len(train_data))
	train(train_data)

if __name__ == "__main__":
	main()