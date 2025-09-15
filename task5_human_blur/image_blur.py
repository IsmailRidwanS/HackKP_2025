import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image
import os

# --- User Input ---
image_path = input("Enter the path to the image you want to blur: ").strip()
output_dir = input("Enter the folder to save the blurred image (default: output_perfect): ").strip()
if not output_dir:
    output_dir = "output_perfect"

# --- Load Image ---
image_cv = cv2.imread(image_path)
if image_cv is None:
    print(f"Error: Could not read the image: {image_path}")
    exit()

print(f"Loaded image: {image_path}")

# --- High-Accuracy Mask Generation using rembg ---
print("Generating perfect human mask with 'u2net_human_seg' model...")
session = new_session("u2net_human_seg")

image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
output_pil = remove(image_pil, session=session, only_mask=True)
mask = np.array(output_pil)  # Binary mask

# --- Applying the Blur ---
if np.any(mask):
    print("Applying strong blur with feathered edges...")

    blur_kernel_size = (151, 151)
    blurred_image_full = cv2.GaussianBlur(image_cv, blur_kernel_size, 0)

    # Feather the mask for smooth blending
    feathered_mask = cv2.GaussianBlur(mask, (21, 21), 0)
    alpha = feathered_mask.astype(float) / 255.0
    alpha_3ch = np.stack([alpha]*3, axis=-1)

    final_image = (alpha_3ch * blurred_image_full) + ((1 - alpha_3ch) * image_cv)
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)
else:
    print("No human detected by the model.")
    final_image = image_cv

# --- Save the Final Image ---
os.makedirs(output_dir, exist_ok=True)
base_name = os.path.basename(image_path)
name, ext = os.path.splitext(base_name)
output_path = os.path.join(output_dir, f"{name}_perfect_feathered_blur{ext}")

cv2.imwrite(output_path, final_image)
print(f"✅ Successfully saved perfectly blurred image to: {output_path}")
