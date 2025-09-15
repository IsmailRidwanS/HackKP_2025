import os
import json
from PIL import Image, ExifTags

def serialize_exif_value(value):
    """
    Convert non-serializable EXIF values (like IFDRational, bytes, etc.)
    into JSON-serializable types (str, float, int).
    """
    if isinstance(value, bytes):
        try:
            return value.decode(errors="ignore")  # try decoding to string
        except:
            return str(value)
    elif hasattr(value, "numerator") and hasattr(value, "denominator"):  
        # Handle IFDRational
        return float(value) if value.denominator != 0 else None
    elif isinstance(value, (list, tuple)):
        return [serialize_exif_value(v) for v in value]
    elif isinstance(value, dict):
        return {str(k): serialize_exif_value(v) for k, v in value.items()}
    else:
        return value if isinstance(value, (str, int, float, type(None))) else str(value)

def extract_metadata(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return {"file": os.path.basename(image_path), "exif": None}
        
        exif = {}
        for tag, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            exif[tag_name] = serialize_exif_value(value)
        
        return {"file": os.path.basename(image_path), "exif": exif}
    except Exception as e:
        return {"file": os.path.basename(image_path), "error": str(e)}

def main():
    input_dir = "./sample"
    output_file = "metadata.json"
    
    results = []
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            filepath = os.path.join(input_dir, filename)
            print(f"Extracting metadata from {filename}...")
            metadata = extract_metadata(filepath)
            results.append(metadata)
    
    # Save results to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Metadata extraction completed! Saved to {output_file}")

if __name__ == "__main__":
    main()
