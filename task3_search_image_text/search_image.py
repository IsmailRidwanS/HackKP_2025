import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------
# 1. Auto-detect dataset folder
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(script_dir, "dataset")  # folder should be named 'dataset' in same dir
if not os.path.isdir(dataset_path):
    raise FileNotFoundError(f"'dataset' folder not found in {script_dir}")

labels_csv = os.path.join(script_dir, "labels.csv")
if not os.path.isfile(labels_csv):
    raise FileNotFoundError(f"'labels.csv' not found in {script_dir}")

# -------------------------
# 2. Load CSV
# -------------------------
df = pd.read_csv(labels_csv)

# -------------------------
# 3. Prepare lowercase mapping of available files
# -------------------------
available_files = {f.lower(): f for f in os.listdir(dataset_path)}

# -------------------------
# 4. User input for search keyword
# -------------------------
query = input("Enter search keyword (e.g., cane, cavallo, elefante, farfalla, gallina): ").strip().lower()

# -------------------------
# 5. Find matching images in CSV
# -------------------------
matches = df[df['label'].str.lower() == query]['filename'].tolist()

if not matches:
    print(f"No images found for '{query}'")
else:
    print(f"Found {len(matches)} images for '{query}'")

    # -------------------------
    # 6. Display top-5 matches
    # -------------------------
    plt.figure(figsize=(15, 5))
    displayed = 0

    for fname in matches:
        if displayed >= 5:  # limit to top 5
            break
        fname_lower = fname.lower()
        if fname_lower in available_files:
            img_path = os.path.join(dataset_path, available_files[fname_lower])
            img = Image.open(img_path)
            plt.subplot(1, 5, displayed + 1)
            plt.imshow(img)
            plt.title(query)
            plt.axis("off")
            displayed += 1
        else:
            print(f"Warning: File {fname} not found in {dataset_path}!")

    if displayed == 0:
        print("No images could be displayed due to missing files.")
    else:
        plt.tight_layout()
        plt.show()
