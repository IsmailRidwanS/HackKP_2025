import os
import pandas as pd

dataset_path = "raw_dataset"

# Prepare CSV rows
rows = []

# Each subfolder is a label
for label in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, label)
    if os.path.isdir(folder_path):
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({"filename": fname, "label": label})

# Save to CSV
df = pd.DataFrame(rows)
df.to_csv("labels.csv", index=False)
print("labels.csv created with", len(df), "rows")

