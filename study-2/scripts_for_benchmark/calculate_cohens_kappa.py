import pandas as pd
from sklearn.metrics import cohen_kappa_score

def calculate_weighted_kappa_from_single_file(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Define dimension names and corresponding annotator columns
    dimensions = {
        "conciseness": ("conciseness_final", "model_conciseness"),
        "fluency": ("fluency_final", "model_fluency"),
        "content_adequacy": ("content_adequacy_final", "model_content_adequacy")
    }

    # Compute Cohen's Kappa with quadratic weights
    kappa_results = {}
    for attribute, (col1, col2) in dimensions.items():
        if col1 in df.columns and col2 in df.columns:
            kappa_quadratic = cohen_kappa_score(df[col1], df[col2], weights="quadratic")
            kappa_results[attribute] = {
                "Quadratic Weighted Kappa": kappa_quadratic
            }
        else:
            kappa_results[attribute] = {
                "Quadratic Weighted Kappa": "Missing columns"
            }

    return kappa_results

# === Example usage ===
file_path = "/home/user/projects/prompt-project/SIDE_p/scripts_and_data/training_SIDE/evaluation_and_statistical_tests/benchmark-gpt-evaluation/final-user-metrics-test-randomized-new-model.csv"
results = calculate_weighted_kappa_from_single_file(file_path)

# === Print results ===
print("Model: Manual Annotations\n")
for attr, scores in results.items():
    print(f"{attr.replace('_', ' ').title()} Kappa Scores:")
    print(f"  Quadratic Weighted Kappa: {scores['Quadratic Weighted Kappa']:.4f}\n")