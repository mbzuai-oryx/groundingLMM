import argparse
import os
import numpy as np
import json
from tqdm import tqdm
from multiprocessing.pool import Pool

MODEL_NAMES = ['co_detr', 'eva-02-01', 'eva-02-02', 'ov_sam', 'grit', 'owl_vit', 'pomp', 'ram', 'tag2text', 'landmark']


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--image_dir_path", required=True)
    parser.add_argument("--predictions_dir_path", required=True, help="Path to the predictions directory.")
    parser.add_argument("--output_dir_path", required=False,
                        default="predictions/level-1-raw")

    parser.add_argument("--num_processes", required=False, type=int, default=32)

    args = parser.parse_args()

    return args


def self_nms_with_score_filter(predictions, score_threshold=0.1, iou_threshold=0.9):
    # Filter out the detections with score less than the threshold
    detections = [det for det in predictions if det["score"] >= score_threshold]

    # Sort the detections based on the confidence score
    detections = sorted(detections, key=lambda x: x["score"], reverse=True)

    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / max(float(boxAArea + boxBArea - interArea), 1e-5)
        return iou

    nms_detections = []
    while len(detections):
        # Take the detection with the highest score
        current_detection = detections.pop(0)
        nms_detections.append(current_detection)

        # Compute its overlap with all the other detections
        # If the overlap is higher than the IOU threshold, discard the detection
        detections = [
            det for det in detections if compute_iou(current_detection["bbox"], det["bbox"]) <= iou_threshold
        ]

    return nms_detections


def remove_segmentation_masks(predictions):
    for pred in predictions:
        del pred["sam_mask"]
    return predictions


def worker(args, image_names):
    for image_name in tqdm(image_names):
        output_json_path = f"{args.output_dir_path}/{image_name}.json"
        if os.path.exists(output_json_path):
            continue
        image_name_with_extension = f"{image_name}.jpg"  # TODO: Handle other extensions
        all_data = {image_name_with_extension: {}}
        for model in MODEL_NAMES:
            json_file_path = f"{args.predictions_dir_path}/{model}/{image_name}.json"
            if not os.path.exists(json_file_path):
                continue
            with open(json_file_path, 'r'):
                try:
                    json_image_data = json.load(open(json_file_path, 'r'))
                except Exception as e:
                    print(f"Exception while reading {json_file_path}")
                    continue
                if model in json_image_data[image_name_with_extension].keys():
                    json_image_data = json_image_data[image_name_with_extension][model]
                else:
                    json_image_data = []
            if model in ['eva-02-01', 'eva-02-02']:
                json_image_data = remove_segmentation_masks(json_image_data)

            if not model in ['ram', 'tag2text', 'landmark']:  # Detection model
                all_data[image_name_with_extension][model] = self_nms_with_score_filter(json_image_data) \
                    if json_image_data else json_image_data
            else:  # Tags model
                all_data[image_name_with_extension][model] = json_image_data

        # Save the level-1 json
        with open(output_json_path, 'w') as f:
            json.dump(all_data, f)


def split_list(input_list, n):
    """Split a list into 'n' parts using numpy."""
    arrays = np.array_split(np.array(input_list), n)
    return [arr.tolist() for arr in arrays]


def main():
    args = parse_args()  # Parse the arguments
    os.makedirs(args.output_dir_path, exist_ok=True)  # Create the directory to save the outputs

    # Get all image names and prepare the task-args
    all_image_names = os.listdir(args.image_dir_path)
    all_image_names = [image_name[:-4] for image_name in all_image_names]
    all_tasks_image_names_list = split_list(all_image_names, n=args.num_processes)
    task_args = [(args, task_image_names) for task_image_names in all_tasks_image_names_list]

    # Use a pool of workers to process the files in parallel.
    with Pool() as pool:
        pool.starmap(worker, task_args)


if __name__ == "__main__":
    main()
