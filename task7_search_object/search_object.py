import os
import glob
from ultralytics import YOLO
from collections import Counter

def find_specific_objects(query_image_path, dataset_path):
    """
    Uses YOLOv8 to find specific objects from a query image in a larger dataset.
    """
    # Load a large, accurate YOLOv8 detection model
    print("Loading high-accuracy YOLOv8 detection model...")
    model = YOLO('yolov8x.pt') # 'x' is for extra-large, the most accurate version

    # --- Step 1: Analyze the query image to find what we're looking for ---
    print(f"\nAnalyzing query image: {os.path.basename(query_image_path)}")
    try:
        query_results = model(query_image_path, verbose=False) # verbose=False keeps the output clean
    except Exception as e:
        print(f"❌ Error analyzing query image: {e}")
        return

    # Get the names of all unique objects detected in the query image
    query_objects = set()
    if query_results:
        for result in query_results:
            for box in result.boxes:
                class_name = model.names[int(box.cls)]
                query_objects.add(class_name)

    if not query_objects:
        print("Could not detect any known objects in the query image. Stopping.")
        return

    print(f"--> Objects to search for: {', '.join(query_objects)}")

    # --- Step 2: Search for these objects in the dataset ---
    print(f"\nSearching for these objects in the dataset at: {dataset_path}")
    image_paths = glob.glob(os.path.join(dataset_path, '*.jpg')) + \
                  glob.glob(os.path.join(dataset_path, '*.png')) + \
                  glob.glob(os.path.join(dataset_path, '*.jpeg'))
    
    matches_found = []

    for i, path in enumerate(image_paths, 1):
        print(f"Processing image {i}/{len(image_paths)}: {os.path.basename(path)}...")
        
        # Skip the query image itself if it's in the dataset
        try:
            if os.path.samefile(query_image_path, path):
                continue
        except FileNotFoundError:
             # This can happen if one of the paths has a strange character
             pass


        dataset_img_results = model(path, verbose=False)
        
        # Get the set of objects in the current dataset image
        current_objects = set()
        if dataset_img_results:
            for result in dataset_img_results:
                for box in result.boxes:
                    current_objects.add(model.names[int(box.cls)])
        
        # Check if there is any overlap between query objects and current image objects
        if query_objects.intersection(current_objects):
            matches_found.append(path)
            print(f"  [MATCH FOUND] Image contains: {', '.join(current_objects)}")

    # --- Step 3: Display the results ---
    print("\n--- Search Complete ---")
    if matches_found:
        print("The following images contain the same object(s) as the query image:")
        for match_path in matches_found:
            print(f" - {match_path}")
    else:
        print("No other images were found containing the specified object(s).")

def main():
    """
    Gets input paths from the user and runs the object search.
    """
    # Get query image path from user
    query_image_path = input("Enter the path to your query image: ").strip()
    if not os.path.isfile(query_image_path):
        print(f"❌ Error: The query file '{query_image_path}' does not exist.")
        return

    # Get dataset folder path from user
    dataset_path = input("Enter the path to the dataset folder to search in: ").strip()
    if not os.path.isdir(dataset_path):
        print(f"❌ Error: The dataset folder '{dataset_path}' does not exist.")
        return

    # Call the main logic function
    find_specific_objects(query_image_path, dataset_path)


if __name__ == '__main__':
    main()