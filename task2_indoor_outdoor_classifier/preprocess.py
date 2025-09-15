import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------
# CONFIG
# -------------------------------
dataset_dir = r"D:\vscode\hackp_2025\task2_indoor_outdoor_classifier\images"
output_dir = r"D:\vscode\hackp_2025\task2_indoor_outdoor_classifier\preprocessed"
os.makedirs(output_dir, exist_ok=True)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# -------------------------------
# COLLECT IMAGE PATHS AND LABELS
# -------------------------------
data = []
classes = sorted(os.listdir(dataset_dir))
for cls in classes:
    cls_folder = os.path.join(dataset_dir, cls)
    if not os.path.isdir(cls_folder):
        continue
    for img_name in os.listdir(cls_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(cls_folder, img_name)
            data.append([img_path, cls])

print(f"Total images found: {len(data)}")
df = pd.DataFrame(data, columns=['image_path', 'label'])

# -------------------------------
# SPLIT DATA
# -------------------------------
train_val, test = train_test_split(df, test_size=test_ratio, stratify=df['label'], random_state=42)
train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio + val_ratio), stratify=train_val['label'], random_state=42)

splits = {'train': train, 'val': val, 'test': test}

# -------------------------------
# CREATE FOLDERS AND COPY IMAGES
# -------------------------------
for split_name, split_df in splits.items():
    for cls in classes:
        cls_dir = os.path.join(output_dir, split_name, cls)
        os.makedirs(cls_dir, exist_ok=True)
    
    for idx, row in split_df.iterrows():
        src = row['image_path']
        dst = os.path.join(output_dir, split_name, row['label'], os.path.basename(src))
        shutil.copy2(src, dst)

# -------------------------------
# SAVE CSV FILES
# -------------------------------
for split_name, split_df in splits.items():
    csv_path = os.path.join(output_dir, f"{split_name}.csv")
    split_df.to_csv(csv_path, index=False)

print("Preprocessing complete!")
print(f"Images and CSVs stored in: {output_dir}")
