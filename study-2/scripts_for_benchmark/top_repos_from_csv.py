import pandas as pd

# Load the CSV file
df = pd.read_csv("./results-new.csv")  # Adjust path if needed

# Sort by stargazers in descending order
sorted_df = df.sort_values(by='stargazers', ascending=False)

# Convert repo names (like "OWNER/REPO") to full GitHub URLs
repo_urls = sorted_df['name'].apply(lambda x: f"https://www.github.com/{x}")

# Save to a text file
repo_urls.to_csv("top_repo_urls_by_stars-new.txt", index=False, header=False)