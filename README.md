# cosmos-logo-text-detect

## Zero-Shot Object Detection with Text Search
This Python script performs zero-shot object detection on a target image using a query image and additionally searches for specified text within the target image using OCR (Optical Character Recognition). The code utilizes the Hugging Face Transformers library for the zero-shot object detection model and the EasyOCR library for text recognition. Object detection is performed based on a pre-trained model checkpoint (google/owlvit-base-patch32). Adjustments to the threshold for object detection confidence can be made by modifying the scores >= 0.4 condition.

## Requirements
- Python 3.x
- PIL (Python Imaging Library)
- torch (PyTorch)
- transformers (Hugging Face Transformers)
- easyocr
- Pre-trained model checkpoint: google/owlvit-base-patch32


## Installation
pip install -r requirements.txt

## Usage
1. Provide the paths to the target image (target_image_path) and the query image (query_image_path).
2. Specify the text you want to search for in the target image (text_to_search).
3. Call the choose_box function with the provided arguments.
4. The function returns a dictionary containing the chosen bounding box coordinates (chosen_box) and the coordinates of the found text in the target image (text_coordinates).

