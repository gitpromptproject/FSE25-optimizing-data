'''import pandas as pd
import numpy as np
import torch
import re
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def infersent_encoding_array_fix(dataframe):
    """InferSent implementation with array shape fix"""
    
    try:
        from infersent.models import InferSent
        
        model_version = 1
        model_path = "infersent/encoder/infersent1.pkl"
        glove_path = 'infersent/GloVe/glove.840B.300d.txt'
        
        print("Loading InferSent model...")
        
        # Model parameters for version 1
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
        
        print("Building InferSent vocabulary...")
        model.build_vocab_k_words(K=100000)
        print("InferSent setup complete!")
        
        def prepare_text_for_infersent(text):
            """Prepare text specifically for InferSent encoding"""
            if not text or pd.isna(text):
                return "empty text description"
            
            text = str(text).strip()
            
            # Basic cleaning
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Ensure it's not empty
            if not text:
                return "empty text description"
            
            # Ensure minimum length for InferSent
            words = text.split()
            if len(words) < 3:
                text = text + " function description here"
            
            return text
        
        cosine_score_dict = []
        failed_count = 0
        
        print("Processing text pairs...")
        for idx, (ref, pred) in enumerate(tqdm(zip(dataframe['originalComment'], dataframe['codeComment']))):
            try:
                # Prepare texts
                ref_clean = prepare_text_for_infersent(ref)
                pred_clean = prepare_text_for_infersent(pred)
                
                # Debug first few pairs
                if idx < 3:
                    print(f"\nDEBUG Row {idx}:")
                    print(f"  Ref: '{ref_clean}'")
                    print(f"  Pred: '{pred_clean}'")
                
                # CRITICAL: Encode each sentence separately to avoid shape issues
                sentences = [ref_clean, pred_clean]
                
                # Try encoding with explicit parameters to avoid array issues
                try:
                    embeddings = model.encode(
                        sentences, 
                        bsize=1,  # Process one at a time to avoid batch issues
                        tokenize=False, 
                        verbose=False
                    )
                    
                    if idx < 3:
                        print(f"  Embeddings shape: {embeddings.shape}")
                        print(f"  Embeddings type: {type(embeddings)}")
                    
                    # Ensure embeddings are numpy array
                    if not isinstance(embeddings, np.ndarray):
                        embeddings = np.array(embeddings)
                    
                    # Check shape
                    if embeddings.shape[0] != 2:
                        if idx < 3:
                            print(f"  ERROR: Wrong embedding shape: {embeddings.shape}")
                        cosine_score_dict.append(0.0)
                        failed_count += 1
                        continue
                    
                    # Calculate cosine similarity using proper indexing
                    emb1 = embeddings[0:1]  # Keep as 2D array
                    emb2 = embeddings[1:2]  # Keep as 2D array
                    
                    if idx < 3:
                        print(f"  Emb1 shape: {emb1.shape}")
                        print(f"  Emb2 shape: {emb2.shape}")
                    
                    # Calculate cosine similarity
                    css = cosine_similarity(emb1, emb2)[0][0]
                    
                    if idx < 3:
                        print(f"  Cosine similarity: {css}")
                    
                    # Validate result
                    if np.isnan(css) or np.isinf(css):
                        cosine_score_dict.append(0.0)
                        failed_count += 1
                    else:
                        cosine_score_dict.append(float(css))
                        
                except Exception as encoding_error:
                    if idx < 5:
                        print(f"  Encoding error: {str(encoding_error)[:100]}")
                    
                    # Try alternative: encode one sentence at a time
                    try:
                        emb1 = model.encode([ref_clean], bsize=1, tokenize=False, verbose=False)
                        emb2 = model.encode([pred_clean], bsize=1, tokenize=False, verbose=False)
                        
                        if not isinstance(emb1, np.ndarray):
                            emb1 = np.array(emb1)
                        if not isinstance(emb2, np.ndarray):
                            emb2 = np.array(emb2)
                        
                        # Ensure 2D shape
                        if emb1.ndim == 1:
                            emb1 = emb1.reshape(1, -1)
                        if emb2.ndim == 1:
                            emb2 = emb2.reshape(1, -1)
                        
                        css = cosine_similarity(emb1, emb2)[0][0]
                        
                        if np.isnan(css) or np.isinf(css):
                            cosine_score_dict.append(0.0)
                            failed_count += 1
                        else:
                            cosine_score_dict.append(float(css))
                            
                    except Exception as alt_error:
                        if idx < 5:
                            print(f"  Alternative encoding also failed: {str(alt_error)[:100]}")
                        cosine_score_dict.append(0.0)
                        failed_count += 1
                
            except Exception as e:
                if idx < 5:
                    print(f"Row {idx} outer error: {str(e)[:100]}")
                cosine_score_dict.append(0.0)
                failed_count += 1
        
        dataframe['InferSent_CS'] = cosine_score_dict
        success_count = len(cosine_score_dict) - failed_count
        print(f"\nInferSent completed. Successful: {success_count}/{len(cosine_score_dict)}, Failed: {failed_count}")
        
        # Print some sample scores to verify they look reasonable
        non_zero_scores = [x for x in cosine_score_dict if x > 0]
        if non_zero_scores:
            print(f"Sample successful scores: {non_zero_scores[:5]}")
            print(f"Score range: {min(non_zero_scores):.3f} to {max(non_zero_scores):.3f}")
        else:
            print("No successful scores - all failed!")
        
    except Exception as e:
        print(f"InferSent setup failed: {e}")
        dataframe['InferSent_CS'] = [0.0] * len(dataframe)
    
    return dataframe


def test_individual_encoding():
    """Test InferSent encoding step by step"""
    
    print("=== INDIVIDUAL ENCODING TEST ===")
    
    try:
        from infersent.models import InferSent
        
        model_version = 1
        model_path = "infersent/encoder/infersent1.pkl"
        glove_path = 'infersent/GloVe/glove.840B.300d.txt'
        
        params_model = {
            'bsize': 64, 
            'word_emb_dim': 300, 
            'enc_lstm_dim': 2048,
            'pool_type': 'max', 
            'dpout_model': 0.0, 
            'version': model_version
        }
        
        model = InferSent(params_model)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.set_w2v_path(glove_path)
        model.build_vocab_k_words(K=100000)
        
        # Test sentences from your data
        test_sentences = [
            "Link a component ID to its producing atom for DAG traceability.",
            "Register a component producer if the atom is recognized, logging the process."
        ]
        
        print(f"Test sentences: {test_sentences}")
        
        # Test different encoding approaches
        print("\n--- Approach 1: Batch encoding ---")
        try:
            embeddings = model.encode(test_sentences, bsize=2, tokenize=False, verbose=False)
            print(f"Success! Shape: {embeddings.shape}, Type: {type(embeddings)}")
            print(f"Embedding 1 shape: {embeddings[0].shape}")
            print(f"Embedding 2 shape: {embeddings[1].shape}")
            
            css = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
            print(f"Cosine similarity: {css}")
            
        except Exception as e:
            print(f"Batch encoding failed: {e}")
        
        print("\n--- Approach 2: Individual encoding ---")
        try:
            emb1 = model.encode([test_sentences[0]], bsize=1, tokenize=False, verbose=False)
            emb2 = model.encode([test_sentences[1]], bsize=1, tokenize=False, verbose=False)
            
            print(f"Emb1 shape: {emb1.shape}, type: {type(emb1)}")
            print(f"Emb2 shape: {emb2.shape}, type: {type(emb2)}")
            
            css = cosine_similarity(emb1, emb2)[0][0]
            print(f"Cosine similarity: {css}")
            
        except Exception as e:
            print(f"Individual encoding failed: {e}")
        
        print("\n--- Approach 3: Single sentences ---")
        try:
            emb1 = model.encode(test_sentences[0], bsize=1, tokenize=False, verbose=False)
            emb2 = model.encode(test_sentences[1], bsize=1, tokenize=False, verbose=False)
            
            print(f"Emb1 shape: {emb1.shape}, type: {type(emb1)}")
            print(f"Emb2 shape: {emb2.shape}, type: {type(emb2)}")
            
            # Reshape if needed
            if emb1.ndim == 1:
                emb1 = emb1.reshape(1, -1)
            if emb2.ndim == 1:
                emb2 = emb2.reshape(1, -1)
                
            css = cosine_similarity(emb1, emb2)[0][0]
            print(f"Cosine similarity: {css}")
            
        except Exception as e:
            print(f"Single sentence encoding failed: {e}")
            
    except Exception as e:
        print(f"Test setup failed: {e}")


def main():
    # First, test individual encoding to see what works
    test_individual_encoding()
    
    print("\n" + "="*50)
    
    # Load your data
    df = pd.read_csv("annotation_All.csv")
    df = df.rename(columns={
        "GT_summary": "originalComment", 
        "generated_summary": "codeComment",
        "input_code": "codeFunctions"
    })
    
    print("Testing on 5 samples with detailed debugging...")
    sample_df = df.head(5).copy()
    result_df = infersent_encoding_array_fix(sample_df)

if __name__ == "__main__":
    main()'''

import pandas as pd
import numpy as np
import torch
import re
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

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


# Replace your current infersent_encoding function with this:
def infersent_encoding(dataframe):
    """Main InferSent function - calls the working implementation"""
    return infersent_encoding_final(dataframe)


def test_on_full_dataset():
    """Test the working version on your full dataset"""
    
    # Load data
    df = pd.read_csv("annotation_All.csv")
    df = df.rename(columns={
        "GT_summary": "originalComment", 
        "generated_summary": "codeComment",
        "input_code": "codeFunctions"
    })
    
    print(f"Processing {len(df)} rows...")
    
    # Process with working InferSent
    df = infersent_encoding_final(df)
    
    # Save results
    df.to_csv("annotation_with_infersent.csv", index=False)
    print("Results saved to 'annotation_with_infersent.csv'")
    
    return df


if __name__ == "__main__":
    test_on_full_dataset()