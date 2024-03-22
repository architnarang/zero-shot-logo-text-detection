from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
import easyocr
import numpy as np


def choose_box(request):
    request_json = request.get_json(silent=True)
    target_image_path = request_json['target_image_path']
    query_image_path = request_json['query_image_path']
    text_to_search = request_json['text_to_search']
    
    # Check if all required arguments are present
    if not all([target_image_path, query_image_path, text_to_search]):
        return 'Please provide all three strings: target_image_path, query_image_path, text_to_search', 400
    
    # Load the images
    image_target = Image.open(target_image_path).convert("RGB")
    query_image = Image.open(query_image_path).convert("RGB")

    # Load model and processor
    checkpoint = "google/owlvit-base-patch32"
    model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
    processor = AutoProcessor.from_pretrained(checkpoint)

    # Resize the images
    resized_target = image_target
    resized_query = query_image

    inputs = processor(images=resized_target, query_images=resized_query, return_tensors="pt")

    with torch.no_grad():
        outputs = model.image_guided_detection(**inputs)
        target_sizes = torch.tensor([resized_target.size[::-1]])
        results = processor.post_process_image_guided_detection(outputs=outputs, target_sizes=target_sizes)[0]

    # Filter out entries where scores < 0.4 - threshold based on testing
    scores = results["scores"]
    boxes = results["boxes"]
    mask = scores >= 0.4
    filtered_scores = scores[mask]
    filtered_boxes = boxes[mask]

    #draw = ImageDraw.Draw(resized_target)

    # Set the number of top boxes to consider
    n = min(len(filtered_boxes), 4)

    # Find the indices of the top n scores
    top_scores, top_indices = torch.topk(filtered_scores, n)

    # Get the corresponding boxes for the top n scores
    top_boxes = filtered_boxes[top_indices]
    top_scores = top_scores.tolist()

    # Calculate the areas of top n bounding boxes
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in top_boxes]

    # Sort the boxes based on their areas
    sorted_indices = sorted(range(len(areas)), key=lambda k: areas[k])

    # # Draw all bounding boxes
    # for box, score in zip(filtered_boxes.tolist(), filtered_scores.tolist()):
    #     xmin, ymin, xmax, ymax = box
    #     draw.rectangle((xmin, ymin, xmax, ymax), outline="black", width=4)
    #     draw.text((xmin, ymin), f"Score: {score}", fill="black")

    # Draw the bounding box with the least area in red
    chosen_index = sorted_indices[0]
    chosen_box = top_boxes[chosen_index]
    chosen_score = top_scores[chosen_index]
    xmin, ymin, xmax, ymax = chosen_box
    # draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=4)
    # draw.text((xmin, ymin), f"Top Score: {chosen_score}", fill="red")

    # Using EasyOCR to find text in the target image
    reader = easyocr.Reader(['en'])
    result = reader.readtext(np.array(image_target))

    # Search for the text in the OCR output
    text_coordinates = None
    for detection in result:
        if text_to_search in detection[1]:
            text_coordinates = detection[0]

    # Prepare output dictionary
    output_data = {
        "chosen_box": chosen_box.tolist(),
        "text_coordinates": text_coordinates
    }

    # Return the resized target image along with the coordinates of the chosen box and text
    return output_data