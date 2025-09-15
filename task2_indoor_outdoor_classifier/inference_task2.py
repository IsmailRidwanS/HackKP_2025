import torch
from torchvision import models, transforms
from PIL import Image
import os
import torch.nn.functional as F

# -------------------------------
# CONFIG
# -------------------------------
# Get the directory where this script is located to build relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'model', 'door.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define classes (must match training order)
class_names = ['Indoor', 'Outdoor']

# -------------------------------
# MODEL SETUP
# -------------------------------
# Check if the model file exists before trying to load it
if not os.path.exists(model_path):
    print(f"❌ Error: Model file not found at {model_path}")
    print("Please ensure the 'model/door.pth' file exists relative to the script.")
    exit()

num_classes = len(class_names)
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval() # Set the model to evaluation mode

# -------------------------------
# IMAGE TRANSFORM
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
# MAIN INFERENCE FUNCTION
# -------------------------------
def predict_image(image_path):
    """Predicts the class and confidence for a single image."""
    try:
        # Open and transform the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            # Get raw model outputs (logits)
            outputs = model(image_tensor)
            
            # Convert logits to probabilities using softmax
            probabilities = F.softmax(outputs, dim=1)
            
            # Get the top probability and its corresponding class index
            confidence, pred_index = torch.max(probabilities, 1)

            # Get the class name and confidence score
            predicted_class = class_names[pred_index.item()]
            confidence_score = confidence.item() * 100
            
            # Print the result
            print(f"-> Prediction: {predicted_class} with {confidence_score:.2f}% confidence")

    except FileNotFoundError:
        print(f"❌ Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while processing the image: {e}")

if __name__ == "__main__":
    # Get a single image path from the user
    query_path = input("Enter the path to your image: ")
    predict_image(query_path)