import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

def generate_code_length_boxplot(jsonl_file, method="tokens"):
    lengths = []

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if method == "tokens":
                tokens = obj.get("code_tokens", [])
                lengths.append(len(tokens))
            elif method == "lines":
                code = obj.get("input_code", "")
                line_count = code.count("\n") + 1 if code else 0
                lengths.append(line_count)

    # Prepare DataFrame
    df = pd.DataFrame(lengths, columns=["Code Length"])
    print("\n Summary Statistics:\n", df["Code Length"].describe())

    # Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(y="Code Length", data=df)
    plt.title(f"Code Length Distribution (method: {method})")
    plt.ylabel("Number of " + ("Tokens" if method == "tokens" else "Lines"))
    plt.tight_layout()

    # Save or show plot
    plot_file = jsonl_file.replace(".jsonl", f"_{method}_code_length_boxplot.png")
    plt.savefig(plot_file)
    print(f"\nBoxplot saved to: {plot_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate boxplot for code length distribution.")
    parser.add_argument("--file", required=True, help="Path to JSONL file")
    parser.add_argument("--method", choices=["tokens", "lines"], default="tokens",
                        help="Measure code length by 'tokens' (code_tokens) or 'lines' (input_code)")
    args = parser.parse_args()

    generate_code_length_boxplot(args.file, args.method)