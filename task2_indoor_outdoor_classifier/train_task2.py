import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# -------------------------------
# CONFIG
# -------------------------------
data_dir = r"D:\vscode\hackp_2025\task2_indoor_outdoor_classifier\preprocessed"
train_csv = os.path.join(data_dir, 'train.csv')
val_csv = os.path.join(data_dir, 'val.csv')
batch_size = 16       # Adjust to fit GPU memory
num_classes = 2
num_epochs = 20       # Increase for better accuracy
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# -------------------------------
# CUSTOM CSV DATASET
# -------------------------------
class CSVImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.class_to_idx = {label: i for i, label in enumerate(sorted(self.data['label'].unique()))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        label_name = row['label']
        label = self.class_to_idx[label_name]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

# -------------------------------
# TRANSFORMS
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
# DATA LOADERS
# -------------------------------
train_dataset = CSVImageDataset(train_csv, transform=train_transform)
val_dataset = CSVImageDataset(val_csv, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = sorted(train_dataset.class_to_idx.keys())

print(f"Classes: {class_names}")
print(f"Dataset sizes: {dataset_sizes}")

# -------------------------------
# MODEL SETUP
# -------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# -------------------------------
# LOSS AND OPTIMIZER
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------------
# TRAINING LOOP
# -------------------------------
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            loader = train_loader
        else:
            model.eval()
            loader = val_loader

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(loader, desc=f"{phase} phase", ncols=100):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

print("Training complete!")

# -------------------------------
# SAVE MODEL
# -------------------------------
model_dir = os.path.join(data_dir, "model")
os.makedirs(model_dir, exist_ok=True)

model_save_path = os.path.join(model_dir, "door.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved at: {model_save_path}")
