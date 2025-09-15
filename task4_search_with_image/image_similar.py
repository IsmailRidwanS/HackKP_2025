import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import difflib

# -------------------------
# 1. Paths
# -------------------------
dataset_path = "dataset"  # folder with images
labels_csv = "labels.csv"  # your CSV file

# -------------------------
# 2. Load CSV
# -------------------------
df = pd.read_csv(labels_csv)

# -------------------------
# 3. Prepare lowercase mapping of available files
# -------------------------
# {lowercase_filename: actual_filename}
available_files = {f.lower(): f for f in os.listdir(dataset_path)}

# -------------------------
# 4. Fuzzy matching function
# -------------------------
def find_closest_file(fname, available_files):
    fname_lower = fname.lower()
    matches = difflib.get_close_matches(fname_lower, available_files.keys(), n=1, cutoff=0.6)
    if matches:
        return os.path.join(dataset_path, available_files[matches[0]])
    return None

# -------------------------
# 5. User input for search keyword
# -------------------------
query = input("Enter search keyword (e.g., cane, cavallo, elefante, farfalla, gallina): ").strip().lower()

# -------------------------
# 6. Find matching images
# -------------------------
matches = df[df['label'].str.lower() == query]['filename'].tolist()

if not matches:
    print(f"No images found for '{query}'")
else:
    print(f"Found {len(matches)} images for '{query}'")

    # -------------------------
    # 7. Display top-5 matches
    # -------------------------
    plt.figure(figsize=(15, 5))
    displayed = 0
    for fname in matches:
        if displayed >= 5:
            break
        img_path = find_closest_file(fname, available_files)
        if img_path and os.path.isfile(img_path):
            img = Image.open(img_path)
            plt.subplot(1, 5, displayed + 1)
            plt.imshow(img)
            plt.title(os.path.basename(img_path))
            plt.axis("off")
            displayed += 1
        else:
            print(f"Warning: File for '{fname}' not found in dataset!")

    if displayed > 0:
        plt.tight_layout()
        plt.show()
    else:
        print("No images could be displayed.")
