import pandas as pd
import numpy as np
import re
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from infersent.models import InferSent
import nltk
nltk.download('punkt', quiet=True)

# ========== BLEU-1 ==========
def indv_bleu_score(dataframe):
    smoothie = SmoothingFunction().method4
    b1dict = []
    for ref, pred in zip(dataframe['originalComment'], dataframe['codeComment']):
        b1dict.append(sentence_bleu([ref.split()], pred.split(), weights=(1,0,0,0), smoothing_function=smoothie))
    
    dataframe['BLEU-1'] = b1dict
    return dataframe

# ========== ROUGE (using py-rouge library) ==========
def indv_rouge_score(dataframe):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge4', 'rougeL'], use_stemmer=True)
        
        rouge_1_p = []
        rouge_4_r = []
        rouge_w_r = []
        
        for ref, pred in zip(dataframe['originalComment'], dataframe['codeComment']):
            ref = str(ref).strip()
            pred = str(pred).strip()
            
            if pred == '' or ref == '':
                rouge_1_p.append(0.0)
                rouge_4_r.append(0.0)
                rouge_w_r.append(0.0)
                continue
                
            try:
                scores = scorer.score(ref, pred)
                rouge_1_p.append(scores['rouge1'].precision)
                rouge_4_r.append(scores.get('rouge4', type('obj', (object,), {'recall': 0.0})).recall)
                rouge_w_r.append(scores['rougeL'].recall)
            except Exception as e:
                print(f"ROUGE Error for pair: {e}")
                rouge_1_p.append(0.0)
                rouge_4_r.append(0.0)
                rouge_w_r.append(0.0)

        dataframe['ROUGE-1-P'] = rouge_1_p
        dataframe['ROUGE-4-R'] = rouge_4_r
        dataframe['ROUGE-W-R'] = rouge_w_r
        
    except ImportError:
        print("rouge_score not installed. Installing alternative ROUGE implementation...")
        # Fallback to simple n-gram overlap
        rouge_1_p = []
        rouge_4_r = []
        rouge_w_r = []
        
        for ref, pred in zip(dataframe['originalComment'], dataframe['codeComment']):
            ref = str(ref).strip().lower().split()
            pred = str(pred).strip().lower().split()
            
            if not pred or not ref:
                rouge_1_p.append(0.0)
                rouge_4_r.append(0.0)
                rouge_w_r.append(0.0)
                continue
            
            # Simple 1-gram precision
            ref_1grams = set(ref)
            pred_1grams = set(pred)
            if pred_1grams:
                precision_1 = len(ref_1grams.intersection(pred_1grams)) / len(pred_1grams)
            else:
                precision_1 = 0.0
            
            # Simple longest common subsequence for ROUGE-L
            def lcs_length(a, b):
                m, n = len(a), len(b)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if a[i-1] == b[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                return dp[m][n]
            
            lcs_len = lcs_length(ref, pred)
            rouge_l_recall = lcs_len / len(ref) if ref else 0.0
            
            rouge_1_p.append(precision_1)
            rouge_4_r.append(0.0)  # Simplified - set to 0
            rouge_w_r.append(rouge_l_recall)
        
        dataframe['ROUGE-1-P'] = rouge_1_p
        dataframe['ROUGE-4-R'] = rouge_4_r
        dataframe['ROUGE-W-R'] = rouge_w_r
    
    return dataframe

# ========== BERTScore-R ==========
def official_bert_score(dataframe):
    p, r, f1 = score(list(dataframe['codeComment']), list(dataframe['originalComment']), lang='en', rescale_with_baseline=True)
    dataframe['BERTScore-R'] = r.numpy()
    return dataframe

# ========== Sentence-BERT Cosine Similarity ==========
def sentence_bert_encoding(dataframe):
    model = SentenceTransformer('stsb-roberta-large')
    cosine_score_dict = []

    for ref, pred in zip(dataframe['originalComment'], dataframe['codeComment']):
        ref = ref.strip()
        pred = pred.strip()
        if pred == '':
            cosine_score_dict.append(0)
            continue
        data = [ref, pred]
        data_emb = model.encode(data)
        
        css = cosine_similarity(data_emb[0].reshape(1, -1), data_emb[1].reshape(1, -1))[0][0]
        cosine_score_dict.append(css)
    
    dataframe['SentenceBERT_CS'] = cosine_score_dict
    return dataframe

# ========== InferSent Cosine Similarity (Fixed) ==========

def infersent_encoding_final(dataframe):
    """Final working InferSent implementation - individual encoding approach"""
    
    try:
        from infersent.models import InferSent
        
        model_version = 1
        model_path = "infersent/encoder/infersent1.pkl"
        glove_path = 'infersent/GloVe/glove.840B.300d.txt'
        
        # Check files exist
        if not os.path.exists(model_path) or not os.path.exists(glove_path):
            print("InferSent files not found")
            dataframe['InferSent_CS'] = [0.0] * len(dataframe)
            return dataframe
        
        print("Loading InferSent model...")
        
        # Model parameters
        params_model = {
            'bsize': 64, 
            'word_emb_dim': 300, 
            'enc_lstm_dim': 2048,
            'pool_type': 'max', 
            'dpout_model': 0.0, 
            'version': model_version
        }
        
        # Load model
        model = InferSent(params_model)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.set_w2v_path(glove_path)
        
        print("Building vocabulary...")
        model.build_vocab_k_words(K=100000)
        print("InferSent ready!")
        
        def clean_text_for_infersent(text):
            """Minimal text cleaning"""
            if not text or pd.isna(text):
                return "empty description"
            
            text = str(text).strip()
            text = re.sub(r'\s+', ' ', text)  # Fix multiple spaces
            
            if not text:
                return "empty description"
                
            return text
        
        cosine_scores = []
        failed_count = 0
        
        print("Processing sentences...")
        
        for idx, (ref, pred) in enumerate(tqdm(zip(dataframe['originalComment'], dataframe['codeComment']))):
            try:
                # Clean texts
                ref_clean = clean_text_for_infersent(ref)
                pred_clean = clean_text_for_infersent(pred)
                
                # CRITICAL: Use individual encoding (Approach 2 from test)
                # This is the only approach that worked in our tests
                emb1 = model.encode([ref_clean], bsize=1, tokenize=False, verbose=False)
                emb2 = model.encode([pred_clean], bsize=1, tokenize=False, verbose=False)
                
                # Calculate cosine similarity
                css = cosine_similarity(emb1, emb2)[0][0]
                
                # Validate result
                if np.isnan(css) or np.isinf(css):
                    cosine_scores.append(0.0)
                    failed_count += 1
                else:
                    cosine_scores.append(float(css))
                    
            except Exception as e:
                # Simple fallback
                cosine_scores.append(0.0)
                failed_count += 1
                
                # Only print first few errors for debugging
                if failed_count <= 3:
                    print(f"Row {idx} failed: {str(e)[:100]}")
        
        # Store results
        dataframe['InferSent_CS'] = cosine_scores
        
        # Report statistics
        success_count = len(cosine_scores) - failed_count
        non_zero_scores = [x for x in cosine_scores if x > 0]
        
        print(f"\nInferSent Results:")
        print(f"  Successful: {success_count}/{len(cosine_scores)}")
        print(f"  Failed: {failed_count}")
        print(f"  Non-zero scores: {len(non_zero_scores)}")
        
        if non_zero_scores:
            print(f"  Score range: {min(non_zero_scores):.3f} to {max(non_zero_scores):.3f}")
            print(f"  Mean score: {np.mean(non_zero_scores):.3f}")
            print(f"  Sample scores: {non_zero_scores[:5]}")
        
    except ImportError as e:
        print(f"InferSent import failed: {e}")
        dataframe['InferSent_CS'] = [0.0] * len(dataframe)
    except Exception as e:
        print(f"InferSent setup failed: {e}")
        dataframe['InferSent_CS'] = [0.0] * len(dataframe)
    
    return dataframe


def infersent_encoding(dataframe):
    """Main InferSent function - calls the working implementation"""
    return infersent_encoding_final(dataframe)

# ========== InferSent Cosine Similarity (Original Attempt) ==========

'''def infersent_encoding(dataframe):
    try:
        import torch
        import re
        import os
        import string
        from infersent.models import InferSent
        
        model_version = 1
        model_path = "infersent/encoder/infersent%s.pkl" % model_version
        params_model = {'bsize':64, 'word_emb_dim':300, 'enc_lstm_dim':2048, 'pool_type':'max',
                        'dpout_model':0.0, 'version':model_version}
        
        # Check if files exist
        if not os.path.exists(model_path):
            print(f"InferSent model not found at {model_path}")
            dataframe['InferSent_CS'] = [0.0] * len(dataframe)
            return dataframe
        
        glove_path = 'infersent/GloVe/glove.840B.300d.txt'
        if not os.path.exists(glove_path):
            print(f"GloVe embeddings not found at {glove_path}")
            dataframe['InferSent_CS'] = [0.0] * len(dataframe)
            return dataframe
        
        print("Loading InferSent model...")
        model = InferSent(params_model)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.set_w2v_path(glove_path)
        
        print("Building InferSent vocabulary...")
        model.build_vocab_k_words(K=100000)
        print("InferSent setup complete!")
        
        def clean_text_for_infersent(text):
            """Clean text to prevent InferSent encoding errors"""
            if not text or not isinstance(text, str):
                return "empty text"
            
            # Convert to string and strip
            text = str(text).strip()
            
            # Replace newlines and tabs with spaces
            text = re.sub(r'[\n\r\t]', ' ', text)
            
            # Remove or replace problematic characters
            text = re.sub(r'[^\w\s\.\,\?\!\-\'\"]', ' ', text)
            
            # Fix multiple spaces
            text = re.sub(r'\s+', ' ', text)
            
            # Remove leading/trailing spaces
            text = text.strip()
            
            # If text is too short, pad it
            words = text.split()
            if len(words) < 2:
                if len(words) == 1:
                    text = f"{words[0]} function"
                else:
                    text = "empty function description"
            
            # Ensure text is not too long (InferSent has limits)
            words = text.split()
            if len(words) > 50:
                text = ' '.join(words[:50])
            
            return text
        
        cosine_score_dict = []
        failed_pairs = []

        for idx, (ref, pred) in enumerate(tqdm(zip(dataframe['originalComment'], dataframe['codeComment']))):
            try:
                # Clean both texts
                ref_clean = clean_text_for_infersent(ref)
                pred_clean = clean_text_for_infersent(pred)
                
                # Double-check they're not empty
                if not ref_clean or not pred_clean:
                    cosine_score_dict.append(0.0)
                    failed_pairs.append(f"Row {idx}: Empty after cleaning")
                    continue
                
                # Encode with InferSent
                data = [ref_clean, pred_clean]
                
                # Try encoding with error handling
                try:
                    data_emb = model.encode(data, bsize=128, tokenize=False, verbose=False)
                    
                    # Ensure we got valid embeddings
                    if data_emb is None or len(data_emb) != 2:
                        cosine_score_dict.append(0.0)
                        failed_pairs.append(f"Row {idx}: Invalid embeddings")
                        continue
                    
                    # Calculate cosine similarity
                    emb1 = data_emb[0].reshape(1, -1)
                    emb2 = data_emb[1].reshape(1, -1)
                    css = cosine_similarity(emb1, emb2)[0][0]
                    
                    # Ensure similarity is valid
                    if np.isnan(css) or np.isinf(css):
                        cosine_score_dict.append(0.0)
                        failed_pairs.append(f"Row {idx}: Invalid similarity")
                    else:
                        cosine_score_dict.append(float(css))
                    
                except Exception as encoding_error:
                    # If encoding fails, try with even simpler text
                    try:
                        # Ultra-simple fallback
                        ref_simple = ' '.join(re.findall(r'\w+', ref_clean)[:10])
                        pred_simple = ' '.join(re.findall(r'\w+', pred_clean)[:10])
                        
                        if not ref_simple:
                            ref_simple = "function"
                        if not pred_simple:
                            pred_simple = "function"
                        
                        data_simple = [ref_simple, pred_simple]
                        data_emb = model.encode(data_simple, bsize=128, tokenize=False, verbose=False)
                        
                        emb1 = data_emb[0].reshape(1, -1)
                        emb2 = data_emb[1].reshape(1, -1)
                        css = cosine_similarity(emb1, emb2)[0][0]
                        
                        if np.isnan(css) or np.isinf(css):
                            cosine_score_dict.append(0.0)
                        else:
                            cosine_score_dict.append(float(css))
                            
                    except Exception:
                        cosine_score_dict.append(0.0)
                        failed_pairs.append(f"Row {idx}: Both attempts failed")
                        
            except Exception as e:
                cosine_score_dict.append(0.0)
                failed_pairs.append(f"Row {idx}: Outer exception - {str(e)[:50]}")
        
        dataframe['InferSent_CS'] = cosine_score_dict
        successful_scores = sum(1 for x in cosine_score_dict if x > 0)
        print(f"InferSent completed. Non-zero scores: {successful_scores}/{len(cosine_score_dict)}")
        
        if len(failed_pairs) > 0:
            print(f"Failed pairs: {len(failed_pairs)}")
            # Print first few failure reasons for debugging
            for i, failure in enumerate(failed_pairs[:5]):
                print(f"  {failure}")
            if len(failed_pairs) > 5:
                print(f"  ... and {len(failed_pairs) - 5} more")
        
    except ImportError as e:
        print(f"InferSent import failed: {e}")
        print("Make sure models.py is downloaded from Facebook's InferSent repo")
        dataframe['InferSent_CS'] = [0.0] * len(dataframe)
    except Exception as e:
        print(f"InferSent setup failed: {e}")
        dataframe['InferSent_CS'] = [0.0] * len(dataframe)
    
    return dataframe'''
# ========== TF-IDF Cosine (for c_coeff) ==========
def tfidf_vectorizer(dataframe):
    cosine_score_dict = []
    for ref, pred in zip(dataframe['originalComment'], dataframe['codeComment']):
        ref = ref.strip()
        pred = pred.strip()
        if pred == '':
            cosine_score_dict.append(0)
            continue
        data = [ref, pred]
        vect = TfidfVectorizer()
        vector_matrix = vect.fit_transform(data)
        
        css = cosine_similarity(np.asarray(vector_matrix[0].todense()), np.asarray(vector_matrix[1].todense()))[0][0]
        cosine_score_dict.append(css)
   
    dataframe['c_coeff'] = cosine_score_dict
    return dataframe

# ========== CodeT5-plus Cosine Similarity ==========
def codet5_plus_encoding(dataframe):
    # Check if 'codeFunctions' column exists, if not use 'codeComment' or create dummy data
    if 'codeFunctions' not in dataframe.columns:
        print("Warning: 'codeFunctions' column not found. Using 'codeComment' as source text.")
        if 'codeComment' in dataframe.columns:
            dataframe['codeFunctions'] = dataframe['codeComment']
        else:
            print("Error: Neither 'codeFunctions' nor 'codeComment' columns found. Skipping CodeT5+ calculation.")
            dataframe['CodeT5-plus_CS'] = [0.0] * len(dataframe)
            return dataframe
    
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    checkpoint = "Salesforce/codet5p-220m"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

    similarities = []
    for idx, item in tqdm(dataframe.iterrows()):
        sentences = []
        # Use codeFunctions and codeComment
        sentences.append(str(item['codeFunctions']))
        sentences.append(str(item['codeComment']))

        try:
            encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)

            with torch.no_grad():
                model_output = model.encoder(
                    input_ids=encoded_input["input_ids"], 
                    attention_mask=encoded_input["attention_mask"], 
                    return_dict=True
                )
              
            # Perform pooling
            sentence_embeddings = mean_pooling(
                model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            sim = util.pytorch_cos_sim(
                sentence_embeddings[0], sentence_embeddings[1]).item()

            similarities.append(sim)
        except Exception as e:
            print(f"CodeT5+ Error for row {idx}: {e}")
            similarities.append(0.0)
    
    dataframe['CodeT5-plus_CS'] = similarities
    return dataframe

# ========== Main ==========
def main():
    df = pd.read_csv("annotation_All.csv")
    df = df.rename(columns={
        "GT_summary": "originalComment", 
        "generated_summary": "codeComment",
        "input_code": "codeFunctions"  # Map input_code to codeFunctions
    })

    print("******************* Computing BLEU-1 SCORES *******************")
    df = indv_bleu_score(df)
    
    print("******************* Computing ROUGE SCORES *******************")
    df = indv_rouge_score(df)

    print("******************* Computing BERT SCORES *******************")
    df = official_bert_score(df)

    print("******************* Computing Sentence BERT SCORES *******************")
    df = sentence_bert_encoding(df)

    print("******************* Computing InferSent SCORES *******************")
    df = infersent_encoding(df)
    
    print("******************* Computing TF-IDF SCORES *******************")
    df = tfidf_vectorizer(df)

    print("******************* Computing CodeT5+ SCORES *******************")
    df = codet5_plus_encoding(df)

    df.to_csv("annotation_with_metrics.csv", index=False)
    print("Saved to 'annotation_with_metrics.csv'")

if __name__ == "__main__":
    main()