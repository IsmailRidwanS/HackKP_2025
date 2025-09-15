import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# -------------------------
# 1. Load Pretrained ResNet50 (remove last layer)
# -------------------------
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = nn.Sequential(*list(model.children())[:-1])  # remove classification layer
model.eval()

# -------------------------
# 2. Image preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_embedding(image_path, device="cpu"):
    """Extract 2048-dim embedding for one image"""
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img).squeeze().cpu().numpy()
    return embedding

# -------------------------
# 3. Fixed dataset folder
# -------------------------
dataset_path = "dataset"  # fixed dataset folder

if not os.path.isdir(dataset_path):
    raise ValueError(f"Dataset folder '{dataset_path}' does not exist!")

# Build database embeddings once
database = {}
for file in os.listdir(dataset_path):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(dataset_path, file)
        database[file] = get_embedding(path)

if not database:
    raise ValueError("No images found in dataset folder!")

# -------------------------
# 4. Dynamic query input
# -------------------------
query_path = input("Enter query image path: ").strip()
if not os.path.isfile(query_path):
    raise ValueError(f"Query image '{query_path}' does not exist!")

query_emb = get_embedding(query_path)

# -------------------------
# 5. Similarity search (Top-5)
# -------------------------
scores = [(fname, float(cosine_similarity([query_emb], [emb])[0][0]))
          for fname, emb in database.items()]

top5 = sorted(scores, key=lambda x: x[1], reverse=True)[:5]

# -------------------------
# 6. Print results
# -------------------------
print("\nQuery image:", query_path)
print("Top 5 most similar images:")
for fname, sim in top5:
    print(f"   {fname}  (similarity = {sim:.4f})")

# -------------------------
# 7. Display query + top-5 matches
# -------------------------
plt.figure(figsize=(15, 6))

# Query image
plt.subplot(1, 6, 1)
plt.imshow(Image.open(query_path))
plt.title("Query")
plt.axis("off")

# Top-5 similar images
for i, (fname, sim) in enumerate(top5, start=2):
    img_path = os.path.join(dataset_path, fname)
    plt.subplot(1, 6, i)
    plt.imshow(Image.open(img_path))
    plt.title(f"Sim {sim:.2f}")
    plt.axis("off")

plt.tight_layout()
plt.show()
