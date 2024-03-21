# cosmos-logo-text-detect

Zero-Shot Object Detection with Text Search
This Python script performs zero-shot object detection on a target image using a query image and additionally searches for specified text within the target image using OCR (Optical Character Recognition). The code utilizes the Hugging Face Transformers library for the zero-shot object detection model and the EasyOCR library for text recognition. Object detection is performed based on a pre-trained model checkpoint (google/owlvit-base-patch32). Adjustments to the threshold for object detection confidence can be made by modifying the scores >= 0.4 condition.

Requirements
Python 3.x
PIL (Python Imaging Library)
torch (PyTorch)
transformers (Hugging Face Transformers)
easyocr
Pre-trained model checkpoint: google/owlvit-base-patch32

Installation
Ensure all dependencies are installed using pip:
pip install pillow torch transformers easyocr

Usage
Provide the paths to the target image (target_image_path) and the query image (query_image_path).
Specify the text you want to search for in the target image (text_to_search).
Call the choose_box function with the provided arguments.
The function returns a dictionary containing the chosen bounding box coordinates (chosen_box) and the coordinates of the found text in the target image (text_coordinates).

Example usage:
target_image_path = "/path/to/target_image.png"
query_image_path = "/path/to/query_image.png"
text_to_search = "example text"
output = choose_box(target_image_path, query_image_path, text_to_search)
print(output)
