import os
import argparse
import pandas as pd
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from sacrebleu.metrics import CHRF


#def calculate_metrics(df, golden_set_df, column_name):
def calculate_metrics(df):
    # predictions = []
    # references = []

    # for (_, inst_gs), (_, inst_df) in zip(golden_set_df.iterrows(), df.iterrows()):
    #     if inst_gs["count"] == "no":
    #         continue

    #     if column_name == "original":
    #         if inst_gs["original_okay"].lower() == "yes":
    #             references.append(inst_gs[column_name].lower())
    #             predictions.append(inst_df["raw_predictions"].lower())
    #     else:
    #         references.append(inst_gs[column_name].lower())
    #         predictions.append(inst_df["raw_predictions"].lower())

    instance_metrics = {"BLEU-4": [], "ROUGE-L": [], "METEOR": [], "ChrF": []}

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method4
    chrf_metric = CHRF()
    # for pred, ref in tqdm(
    #     zip(predictions, references), total=len(predictions), desc="Calculating metrics"
    # ):
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating metrics"):
        ref = row["target"].strip().lower()
        pred = row["raw_predictions"].strip().lower()

        pred_tokens = word_tokenize(pred)
        ref_tokens = word_tokenize(ref)

        # Custom BLEU-4 calculation
        bleu_score = corpus_bleu(
            [[ref_tokens]], [pred_tokens], smoothing_function=smooth
        )
        instance_metrics["BLEU-4"].append(bleu_score)

        # ROUGE-L score
        rouge_l_score = rouge.score(ref, pred)["rougeL"].fmeasure
        instance_metrics["ROUGE-L"].append(rouge_l_score)

        # METEOR score
        meteor_score_val = meteor_score([ref_tokens], pred_tokens)
        instance_metrics["METEOR"].append(meteor_score_val)

        # CHRF score
        chrf_score_val = chrf_metric.sentence_score(pred, [ref]).score / 100  # scale to [0, 1]
        instance_metrics["ChrF"].append(chrf_score_val)

    return instance_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process predictions and compute metrics."
    )
    parser.add_argument(
        "--prediction-path",
        type=str,
        required=True,
        help="Path to the directory containing prediction files.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="evaluation_metrics.csv",
        help="Output file for saving the evaluation results.",
    )
    parser.add_argument(
        "--gt-label", type=str, default="human", help="Label for ground-truth."
    )

    args = parser.parse_args()

    predictions_path = args.prediction_path
    output_file = args.output_file
    gt_label = args.gt_label

    #output_dir = os.path.dirname(output_file)
    output_dir = output_file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Only process for beam 1
    # beam = 1
    # df = pd.read_csv(os.path.join(predictions_path, f"golden-test-results@{beam}.csv"))
    df = pd.read_csv(predictions_path)
    instance_metrics = calculate_metrics(df)

    #golden_set_df = pd.read_csv("../CoderEval/codereval.csv")
    #instance_metrics = calculate_metrics(df, golden_set_df, gt_label)

    # Save instance-level scores to separate CSV files
    for metric, values in instance_metrics.items():
        instance_df = pd.DataFrame({"Instance": range(len(values)), metric: values})
        #instance_output_file = f"{output_file}_{metric.lower()}.csv"
        basename = os.path.basename(os.path.normpath(output_dir))   
        instance_output_file = os.path.join(output_dir, f"{basename}_{metric.lower()}.csv")
        instance_df.to_csv(instance_output_file, index=False)
        print(f"Instance-level {metric} scores saved to {instance_output_file}")
        
    # Print overall (corpus-level) average scores
    print("\n=== Overall Corpus-Level Scores (ICPC script) ===")
    for metric, values in instance_metrics.items():
        avg_score = sum(values) / len(values) if values else 0
        print(f"{metric}: {avg_score:.4f}")