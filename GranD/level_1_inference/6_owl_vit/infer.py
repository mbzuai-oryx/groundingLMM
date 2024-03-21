import argparse
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch
import os
import json
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import supervision as sv
from typing import List
from torchvision.ops import box_convert
from tqdm import tqdm
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--image_names_txt_path", required=True)
    parser.add_argument("--image_dir_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--tags_dir_path", required=True)

    args = parser.parse_args()

    return args


def get_predictions(text_queries, scores, boxes, labels, score_threshold, output_file_path):
    new_boxes = []
    new_scores = []
    new_classes = []
    for score, box, label in zip(scores, boxes, labels):
        if score < score_threshold:
            continue

        new_boxes.append(box)
        new_scores.append(score)
        new_classes.append(text_queries[label])

    return new_boxes, new_scores, new_classes


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    h, w, _ = image_source.shape

    # Ensure boxes is a tensor
    boxes = torch.tensor(boxes)

    # If boxes are empty, return original image with empty xyxy.
    if boxes.numel() == 0:
        return image_source, []

    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    return annotated_frame, xyxy


def run_owl_vit(model, processor, image_path, text_queries, device):
    image = Image.open(image_path)

    # If grayscale (2D image), add a channel dimension
    # Convert image to numpy array
    image_np = np.array(image)
    if image_np.ndim == 2:
        image_np = np.repeat(image_np[:, :, np.newaxis], 3, axis=2)
        image = Image.fromarray(image_np)

    # Process image and text inputs
    inputs = processor(text=text_queries, images=image, return_tensors="pt").to(device)
    # Set model in evaluation mode
    model = model.to(device)
    model.eval()
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    input_image = np.asarray(image)

    # Threshold to eliminate low probability predictions
    score_threshold = 0.05

    # Get prediction logits
    logits = torch.max(outputs["logits"][0], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()

    # Get prediction labels and boundary boxes
    labels = logits.indices.cpu().detach().numpy()
    boxes = outputs["pred_boxes"][0].cpu().detach().numpy()

    return input_image, text_queries, scores, boxes, labels, score_threshold, image.size


def json_serializable(data):
    if isinstance(data, np.float32):  # if it's a np.float32
        return round(float(data), 2)  # convert to python float
    elif isinstance(data, np.ndarray):  # if it's a np.ndarray
        return data.tolist()  # convert to python list
    else:  # for other types, let it handle normally
        return data


def main():
    args = parse_args()
    image_dir_path = args.image_dir_path
    output_dir_path = args.output_dir_path
    tags_dir_path = args.tags_dir_path

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.makedirs(output_dir_path, exist_ok=True)

    # Load model
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

    all_data = {}
    model_name = "owl_vit"
    processed_image_dir = os.path.join(args.output_dir_path, model_name)
    os.makedirs(processed_image_dir, exist_ok=True)
    processed_images = os.listdir(processed_image_dir)

    start_time = time.time()

    with open(args.image_names_txt_path, 'r') as f:
        image_names = [line.strip() for line in f if line.strip()]

    for image_name in tqdm(image_names):
        output_file_name = f'{image_name[:-4]}.json'
        if output_file_name in processed_images:
            continue
        image_path = f"{image_dir_path}/{image_name}"

        try:
            tag2text_tags = json.load(open(f"{tags_dir_path}/tag2text/{image_name[:-4]}.json", 'r'))
        except Exception as e:
            print(f"Tag2Text exception at image {image_name}")
            tag2text_tags = {image_name: {"tag2text": {"tags": []}}}
        try:
            ram_tags = json.load(open(f"{tags_dir_path}/ram/{image_name[:-4]}.json", 'r'))
        except Exception as e:
            print(f"Ram exception at image {image_name}")
            ram_tags = {image_name: {"ram": {"tags": []}}}

        tag2text_tags = tag2text_tags[image_name]["tag2text"]["tags"]
        ram_tags = ram_tags[image_name]["ram"]["tags"]
        classes = list(set(tag2text_tags + ram_tags))
        classes.append('trees')
        if len(classes) == 0:
            print(f"Skipping image {image_name} as no tags are found.")
            continue

        all_data[image_name] = {}

        input_image, text_queries, scores, boxes, labels, score_threshold, original_image_size = \
            run_owl_vit(model, processor, image_path, classes, device)
        output_file_path = f"{output_dir_path}/{image_name}.jpg"
        new_boxes, new_scores, new_classes = \
            get_predictions(text_queries, scores, boxes, labels, score_threshold, output_file_path)

        annotated_frame, xyxy = annotate(input_image, new_boxes, new_scores, new_classes)

        all_image_predictions = []
        # Check if xyxy is not empty
        if len(xyxy) > 0:
            for j, box in enumerate(xyxy.tolist()):
                prediction = {}
                prediction['bbox'] = [round(float(b), 2) for b in box]
                prediction['score'] = round(new_scores[j], 2)
                prediction['label'] = new_classes[j]
                all_image_predictions.append(prediction)

            all_data[image_name][model_name] = [{k: json_serializable(v) for k, v in prediction.items()} for
                                                prediction in all_image_predictions]

        # Write all_data to a JSON file
        with open(os.path.join(processed_image_dir, output_file_name), 'w') as f:
            json.dump(all_data, f)
        all_data = {}

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- OWLViT Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')


if __name__ == "__main__":
    main()
