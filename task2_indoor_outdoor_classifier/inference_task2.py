import torch
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
model_path = r"D:\vscode\hackp_2025\task2_indoor_outdoor_classifier\model\door.pth"
test_dir = r"D:\vscode\hackp_2025\task2_indoor_outdoor_classifier\sample_input"
output_csv = r"D:\vscode\hackp_2025\task2_indoor_outdoor_classifier\inference_results.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define classes (must match training order)
class_names = ['Indoor', 'Outdoor']

# -------------------------------
# MODEL SETUP
# -------------------------------
num_classes = len(class_names)
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# -------------------------------
# IMAGE TRANSFORM
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
# INFERENCE ON ALL IMAGES IN TEST DIR
# -------------------------------
results = []
with torch.no_grad():
    for img_file in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            outputs = model(image)
            _, pred = torch.max(outputs, 1)
            predicted_class = class_names[pred.item()]

            results.append([img_file, predicted_class])
            print(f"{img_file} → {predicted_class}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# -------------------------------
# SAVE PREDICTIONS
# -------------------------------
df_out = pd.DataFrame(results, columns=["image_name", "predicted_label"])
df_out.to_csv(output_csv, index=False)
print(f"\n✅ Inference complete! Predictions saved to: {output_csv}")
