import argparse
import os
import json
from mmdet.apis import init_detector, inference_detector
import numpy as np
from tqdm import tqdm
import time

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic_light',
               'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard',
               'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--image_names_txt_path", required=True)
    parser.add_argument("--image_dir_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--sam_encoder_version", required=False, default="vit_h")

    parser.add_argument("--ckpt_path", required=True, help="Path to the pretrained checkpoints path.")
    parser.add_argument("--config_file", required=False,
                        default="projects/configs/co_deformable_detr/co_deformable_detr_swin_large_900q_3x_coco.py",
                        help="Config file path.")

    args = parser.parse_args()

    return args


def json_serializable(data):
    if isinstance(data, np.float32):  # if it's a np.float32
        return round(float(data), 2)  # convert to python float
    elif isinstance(data, np.ndarray):  # if it's a np.ndarray
        return data.tolist()  # convert to python list
    else:  # for other types, let it handle normally
        return data


def run_inference(model, img_path):
    return inference_detector(model, img_path)


def main():
    args = parse_args()
    image_dir_path = args.image_dir_path
    output_dir_path = args.output_dir_path
    ckpt_path = args.ckpt_path
    config_file = args.config_file

    os.makedirs(output_dir_path, exist_ok=True)
    os.makedirs(f"{output_dir_path}/co_detr", exist_ok=True)

    # CO-DETR model
    co_detr_model = init_detector(config_file, ckpt_path, device='cuda')
    start_time = time.time()
    with open(args.image_names_txt_path, 'r') as f:
        image_names = [line.strip() for line in f if line.strip()]
    all_data = {}
    for image_name in tqdm(image_names):
        output_json_path = f"{output_dir_path}/co_detr/{image_name[:-4]}.json"
        if os.path.exists(output_json_path):
            continue
        all_data[image_name] = {}
        image_path = f"{image_dir_path}/{image_name}"
        predictions = run_inference(co_detr_model, image_path)
        # Get bounding boxes
        score_thr = 0.0
        bboxes = np.vstack(predictions)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(predictions)
        ]
        labels = np.concatenate(labels)
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        bboxes_int = []
        score_values = []

        for box in bboxes:
            score = box[-1]
            box = [round(float(b), 2) for b in box]
            box = box[:-1]
            bboxes_int.append(box)
            score_values.append(round(score, 2))

        labels = labels[inds]
        label_names = [class_names[label] for label in labels]

        all_image_predictions = []
        for j, box in enumerate(bboxes_int):
            prediction = {}
            prediction['bbox'] = box
            prediction['score'] = round(score_values[j], 2)
            prediction['label'] = label_names[j]
            all_image_predictions.append(prediction)

        all_data[image_name]['co_detr'] = [{k: json_serializable(v) for k, v in prediction.items()} for prediction in
                                           all_image_predictions]
        # Write all_data to a JSON file
        with open(output_json_path, 'w') as f:
            json.dump(all_data, f)
        all_data = {}
    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- codetr Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')


if __name__ == "__main__":
    main()
