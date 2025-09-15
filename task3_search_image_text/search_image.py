import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------
# 1. Paths
# -------------------------
dataset_path = "raw_dataset"  # folder containing your images
labels_csv = "labels.csv"     # your CSV file

# -------------------------
# 2. Load CSV
# -------------------------
df = pd.read_csv(labels_csv)

# -------------------------
# 3. User input for search keyword
# -------------------------
query = input(
    "Enter search keyword (e.g., cane, cavallo, elefante, farfalla, gallina): "
).strip().lower()

# -------------------------
# 4. Find matching images
# -------------------------
matches = df[df['label'].str.lower() == query]['filename'].tolist()

if not matches:
    print(f"No images found for '{query}'")
else:
    print(f"Found {len(matches)} images for '{query}'")

    # -------------------------
    # 5. Display top-5 matches
    # -------------------------
    plt.figure(figsize=(15, 5))
    for i, fname in enumerate(matches[:5], start=1):
        img_path = os.path.join(dataset_path, fname)
        if os.path.isfile(img_path):
            img = Image.open(img_path)
            plt.subplot(1, 5, i)
            plt.imshow(img)
            plt.title(fname)
            plt.axis("off")
        else:
            print(f"Warning: File {img_path} not found!")
    plt.tight_layout()
    plt.show()
