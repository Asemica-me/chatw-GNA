import pandas as pd

data = {
    "Prompt": [1, 2, 3, 4, 5, 6],
    "Consistency": [5, 5, 5, 5, 5, 5],
    "Relevance": [5, 5, 5, 5, 5, 0],
    "Fluency": [5, 5, 5, 5, 4, 5],
    "Completeness": [5, 5, 5, 5, 3, 2]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save as CSV
csv_filename = "outputs/prompt_evaluations.csv"
df.to_csv(csv_filename, index=False)

# Return the file path
csv_filename
