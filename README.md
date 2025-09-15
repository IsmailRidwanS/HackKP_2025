# HackKP_2025

This repository contains a collection of Python scripts designed for analyzing digital media as part of the HACKP-2025 competition.

## 🧑‍💻 Team Information

- **Team Leader**: Ismail Ridwan S
- **Team Members**:
    - Danish A G
    - Sabaresh C
    - Kumaresan H
- **College**: Chennai Institute of Technology and Applied Research
---
## ✅ Tasks Attempted

This project successfully implements the following five tasks:
1. **Metadata Analysis**: A script to extract and display EXIF metadata from an image file.
2. **Indoor/Outdoor Classifier**: A machine learning model that classifies an image as being taken indoors or outdoors.
3. **Text-Based Image Search**: A system to retrieve images from a dataset based on a natural language text query.
4. **Image-Based Search**: A content-based image retrieval (CBIR) system to find visually similar images to a given query image.
5. **Human Blurring**: A tool to automatically detect and blur human figures in an image for anonymization.

---

## 🚀 Setup & Instructions

### General Setup

First, clone the repository and install the necessary dependencies. It's highly recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/IsmailRidwanS/HackKP_2025.git
cd HackKP_2025

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install all required libraries
pip install -r requirements.txt
```

### How to Run Each Task

Below are the specific instructions to run each task from the root directory of the project.

#### Task 1: Metadata Analysis

- **Description**: Extracts and prints EXIF metadata from a given image.
- **Dataset**: Uses a sample image located at `task1_metadata_analysis/sample/image.jpg`.
- **Command**:
    ```
    python task1_metadata_analysis/main.py task1_metadata_analysis/sample/image.jpg
    ```
#### Task 2: Indoor/Outdoor Classifier
- **Description**: Classifies a sample image as "indoor" or "outdoor" using a pre-trained model.
- **Dataset**: The model was trained on the "Indoor Outdoor Scene Recognition" dataset. Inference is performed on a sample image located at `task2_indoor_outdoor_classifier/sample_input/test_image.jpg`.
- **Command**:
    ```bash
    python task2_indoor_outdoor_classifier/inference_task2.py
    ```
#### Task 3: Text-Based Image Search
- **Description**: Searches for images in a dataset using a text query.
- **Dataset**: Utilizes the image dataset located in the `task3_search_image_text/dataset/` directory.
- **Command**   
    ```bash
    python task3_search_image_text/search_image.py 
    ```

#### Task 4: Image-Based Search

- **Description**: Finds images in the dataset that are visually similar to a provided input image.    
- **Dataset**: Searches against the dataset located in the `task4_search_with_image/dataset/` directory.
- **Command**:
    ```bash
      python task4_search_with_image/image_similar.py 
    ```
#### Task 5: Human Blurrin
- **Description**: Detects and blurs any people found in an input image and saves the result.
- **Dataset**: A sample input image is provided at `task5_human_blur/sample_input/test.jpeg`.
- **Command**:

    ```bash
    python task5_human_blur/image_blur.py
    ```
    Task 7: Exact Object Search
        Description: Finds a specific object within a directory of images using a query image and saves visual proof of any matches to an output folder.

        Dataset: A query image provided by the user (e.g., query_image.jpg) and a dataset folder of images to search through (e.g., raw_dataset/).

        Command:
            ```bash
                python search_tool.py
            ```

---


## 🛠️ External Libraries & Models

This project relies on several key external libraries and pre-trained models. All Python libraries are listed in `requirements.txt`.

- **Core Libraries**:
	- Pillow`: For image manipulation and metadata extraction.    
    - `OpenCV (cv2)`: Used for computer vision tasks like object detection.
    - `PyTorch (torch)` & `torchvision`: For building, training, and using deep learning models.
    - `tqdm`: For displaying progress bars.
    - `scikit-learn`: For machine learning utilities.
    - `transformers`: For accessing pre-trained models from Hugging Face.
- **Pre-trained Models**:
    - **Task 2**: A fine-tuned **ResNet** model for image classification.
    - **Task 3 & 4**: A **CLIP (Contrastive Language–Image Pre-training)** model for generating joint text and image embeddings
    - **Task 5**: A pre-trained model like **YOLO (You Only Look Once)** or a similar object detector for identifying human figures.