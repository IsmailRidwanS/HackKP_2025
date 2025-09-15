import os
import json
from PIL import Image, ExifTags
from datetime import datetime

# A dictionary to map numeric codes to human-readable names for common tags
TAG_LOOKUP = {
    'WhiteBalance': {0: 'Auto', 1: 'Manual'},
    'MeteringMode': {0: 'Unknown', 1: 'Average', 2: 'CenterWeightedAverage', 3: 'Spot', 4: 'MultiSpot', 5: 'Pattern', 6: 'Partial', 255: 'Other'},
    'ExposureProgram': {0: 'Not defined', 1: 'Manual', 2: 'Normal program', 3: 'Aperture priority', 4: 'Shutter priority', 5: 'Creative program', 6: 'Action program', 7: 'Portrait mode', 8: 'Landscape mode'},
    'SceneCaptureType': {0: 'Standard', 1: 'Landscape', 2: 'Portrait', 3: 'Night scene'},
    'Flash': {
        0: 'Flash did not fire', 1: 'Flash fired', 5: 'Strobe return light not detected',
        7: 'Strobe return light detected', 9: 'Flash fired, compulsory flash mode',
        13: 'Flash fired, compulsory flash mode, return light not detected',
        15: 'Flash fired, compulsory flash mode, return light detected',
        16: 'Flash did not fire, compulsory flash mode', 24: 'Flash did not fire, auto mode',
        25: 'Flash fired, auto mode', 29: 'Flash fired, auto mode, return light not detected',
        31: 'Flash fired, auto mode, return light detected', 32: 'No flash function',
        65: 'Flash fired, red-eye reduction mode',
        69: 'Flash fired, red-eye reduction mode, return light not detected',
        71: 'Flash fired, red-eye reduction mode, return light detected',
        73: 'Flash fired, compulsory flash mode, red-eye reduction mode',
        77: 'Flash fired, compulsory flash mode, red-eye reduction mode, return light not detected',
        79: 'Flash fired, compulsory flash mode, red-eye reduction mode, return light detected',
        89: 'Flash fired, auto mode, red-eye reduction mode',
        93: 'Flash fired, auto mode, return light not detected, red-eye reduction mode',
        95: 'Flash fired, auto mode, return light detected, red-eye reduction mode'
    }
}

def serialize_exif_value(value):
    """Converts non-serializable EXIF values to JSON-serializable types."""
    if isinstance(value, bytes):
        try:
            return value.partition(b'\0')[0].decode(errors="ignore").strip()
        except Exception:
            return str(value)
    elif hasattr(value, "numerator") and hasattr(value, "denominator"):
        return float(value) if value.denominator != 0 else None
    elif isinstance(value, (list, tuple)):
        return [serialize_exif_value(v) for v in value]
    return value if isinstance(value, (str, int, float, type(None))) else str(value)

def dms_to_dd(dms, ref):
    """Converts GPS Degrees, Minutes, Seconds to Decimal Degrees."""
    if not dms or len(dms) < 3:
        return None
    try:
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])
        dd = degrees + minutes / 60.0 + seconds / 3600.0
        if ref in ['S', 'W']:
            dd *= -1
        return dd
    except (ValueError, TypeError):
        return None

def analyze_image_metadata(image_path):
    """Extracts, cleans, and summarizes metadata from a single image."""
    filename = os.path.basename(image_path)
    summary = {
        "file": filename,
        "camera_make": "Unknown",
        "camera_model": "Unknown",
        "date_taken": None,
        "location": "Not specified",
        "edited": False,
        "editing_software": "None",
        "problems": []
    }
    raw_exif = {}

    try:
        img = Image.open(image_path)
        exif_data = img._getexif()

        if not exif_data:
            summary["problems"].append("No EXIF data found.")
            return summary, raw_exif

        for tag_id, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            raw_exif[tag_name] = serialize_exif_value(value)

        # --- Populate Summary ---
        summary["camera_make"] = raw_exif.get("Make", "Unknown")
        summary["camera_model"] = raw_exif.get("Model", "Unknown")

        software = raw_exif.get("Software", "")
        if software and any(editor in software for editor in ["Photoshop", "GIMP", "Lightroom"]):
            summary["edited"] = True
            summary["editing_software"] = software

        date_str = raw_exif.get("DateTimeOriginal") or raw_exif.get("DateTime")
        if date_str and isinstance(date_str, str):
             try:
                clean_date_str = date_str.strip().replace(":", "-", 2)
                summary["date_taken"] = datetime.strptime(clean_date_str, '%Y-%m-%d %H:%M:%S').isoformat()
             except ValueError:
                summary["problems"].append(f"Invalid date format: {date_str}")
                summary["date_taken"] = date_str

        gps_info = raw_exif.get("GPSInfo")
        if isinstance(gps_info, dict):
            lat_dms = gps_info.get(2)
            lat_ref = gps_info.get(1)
            lon_dms = gps_info.get(4)
            lon_ref = gps_info.get(3)
            
            if lat_dms and lat_ref and lon_dms and lon_ref:
                lat_dd = dms_to_dd(lat_dms, lat_ref)
                lon_dd = dms_to_dd(lon_dms, lon_ref)
                if lat_dd is not None and lon_dd is not None:
                    summary["location"] = f"{lat_dd:.6f}, {lon_dd:.6f}"

        for key, val in raw_exif.items():
            if key in TAG_LOOKUP and isinstance(val, (int, float)):
                 raw_exif[key] = f"{TAG_LOOKUP[key].get(int(val), 'Unknown')} ({val})"

    except Exception as e:
        summary["problems"].append(f"Error processing file: {str(e)}")

    return summary, raw_exif

def main():
    """Main function to handle a single file or a directory with one input."""
    # --- Get single input from the user ---
    input_path = input("Enter the path to an image file OR a directory of images: ")

    # --- Check if the path is a file or a directory ---
    image_paths = []
    supported_extensions = (".jpg", ".jpeg", ".tiff", ".png")

    if os.path.isfile(input_path):
        if input_path.lower().endswith(supported_extensions):
            image_paths.append(input_path)
        else:
            print("❌ Error: The provided file is not a supported image type.")
            return
    elif os.path.isdir(input_path):
        for filename in sorted(os.listdir(input_path)):
            if filename.lower().endswith(supported_extensions):
                image_paths.append(os.path.join(input_path, filename))
    else:
        print(f"❌ Error: The path '{input_path}' is not a valid file or directory.")
        return

    if not image_paths:
        print(f"🤔 No image files found at the specified path.")
        return

    all_summaries = []
    all_raw_data = []

    # Process the list of image paths (1 for a file, many for a directory)
    for filepath in image_paths:
        print(f"Analyzing {os.path.basename(filepath)}...")
        summary, raw_exif = analyze_image_metadata(filepath)
        all_summaries.append(summary)
        all_raw_data.append({"file": os.path.basename(filepath), "exif": raw_exif})

    # Save results to hardcoded filenames
    summary_output_file = "metadata_summary.json"
    raw_output_file = "metadata_raw.json"

    with open(summary_output_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=4, ensure_ascii=False)

    with open(raw_output_file, "w", encoding="utf-8") as f:
        json.dump(all_raw_data, f, indent=4, ensure_ascii=False)

    print("\n✅ Analysis Complete!")
    print(f"- A clean summary has been saved to {summary_output_file}")
    print(f"- The full, raw metadata has been saved to {raw_output_file}")

if __name__ == "__main__":
    main()