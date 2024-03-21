import numpy as np
from PIL import ImageDraw, Image, ImageFont
from tqdm import tqdm
import os
import time
import argparse
import torch
import torch.nn.functional as F
import json
from mmdet.registry import MODELS
from mmengine import Config, print_log
from mmengine.structures import InstanceData
from ext.class_names.lvis_list import LVIS_CLASSES

LVIS_NAMES = LVIS_CLASSES
class_names = ['person', 'helmet', 'motorbike', 'jacket']
class_names = LVIS_NAMES
model_cfg = Config.fromfile('app/configs/sam_r50x16_fpn.py')

model = MODELS.build(model_cfg.model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)
model = model.eval()
model.init_weights()

mean = torch.tensor([123.675, 116.28, 103.53], device=device)[:, None, None]
std = torch.tensor([58.395, 57.12, 57.375], device=device)[:, None, None]
IMG_SIZE = 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Open Vocabulary SAM")

    parser.add_argument("--image_names_txt_path", required=True)
    parser.add_argument("--image_dir_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--sam_annotations_dir", required=True,
                        help="path to the directory containing all sam annotations.")

    args = parser.parse_args()

    return args


def extract_img_feat(img):
    w, h = img.size
    scale = IMG_SIZE / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    img_numpy = np.array(img)

    try:
        img_tensor = torch.tensor(img_numpy, device=device, dtype=torch.float32).permute((2, 0, 1))[None]
        img_tensor = (img_tensor - mean) / std
        img_tensor = F.pad(img_tensor, (0, IMG_SIZE - new_w, 0, IMG_SIZE - new_h), 'constant', 0)
        feat_dict = model.extract_feat(img_tensor)
        img_feat = feat_dict
        if img_feat is not None:
            for k in img_feat:
                if isinstance(img_feat[k], torch.Tensor):
                    img_feat[k] = img_feat[k].to(device)
                elif isinstance(img_feat[k], tuple):
                    img_feat[k] = tuple(v.to(device) for v in img_feat[k])
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print_log(f"CUDA OOM! please try again later", logger='current')
            return None, None, "CUDA OOM, please try again later."
        else:
            raise

    return img_numpy, img_feat


def get_bbox_with_draw(image, bbox, index):
    # Each bbox is a tuple (x, y, w, h)
    point_radius, point_color, box_outline = 5, (237, 34, 13), 2
    myFont = ImageFont.load_default()
    box_color, text_color = (237, 34, 13), (255, 0, 0)
    draw = ImageDraw.Draw(image)
    x, y, w, h = bbox
    draw.rectangle(
        [x, y, x + w, y + h],
        outline=box_color,
        width=box_outline
    )
    # Draw index number
    draw.text((x, y), str(index), fill=text_color, font=myFont)
    xyxy = [x, y, x + w, y + h]

    return image, xyxy


def run_inference(numpy_image, img_feat, selected_bboxes):
    output_img = numpy_image
    h, w = output_img.shape[:2]
    box_points = selected_bboxes
    bbox = (
        min(box_points[0][0], box_points[1][0]),
        min(box_points[0][1], box_points[1][1]),
        max(box_points[0][0], box_points[1][0]),
        max(box_points[0][1], box_points[1][1]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_bbox = torch.tensor(bbox, dtype=torch.float32, device=device)
    prompts = InstanceData(
        bboxes=input_bbox[None],
    )

    try:
        masks, cls_pred = model.extract_masks(img_feat, prompts)
        masks = masks[0, 0, :h, :w]
        masks = masks > 0.5

        cls_pred = cls_pred[0][0]
        scores, indices = torch.topk(cls_pred, 1)
        scores, indices = scores.tolist(), indices.tolist()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print_log(f"CUDA OOM! please try again later", logger='current')
            return None, None, "CUDA OOM, please try again later."
        else:
            raise
    names = []

    for ind in indices:
        names.append(class_names[ind].replace('_', ' '))

    cls_info = ""
    for name, score in zip(names, scores):
        cls_info += "{} ({:.2f})\n".format(name, score)

    rgb_shape = tuple(list(masks.shape) + [3])
    color = np.zeros(rgb_shape, dtype=np.uint8)
    color[masks] = np.array([97, 217, 54])

    return cls_info, names[0], scores[0]


def json_serializable(data):
    if isinstance(data, np.float32):  # if it's a np.float32
        return round(float(data), 2)  # convert to python float
    elif isinstance(data, np.ndarray):  # if it's a np.ndarray
        return data.tolist()  # convert to python list
    else:  # for other types, let it handle normally
        return data


def filter_sam_boxes(annotations, threshold):
    filtered_annotations = []
    for ann in annotations:
        bbox = ann['bbox']
        area = bbox[2] * bbox[3]
        image_area = ann['segmentation']['size'][0] * ann['segmentation']['size'][1]
        area_ratio = area / image_area
        if area_ratio * 100 > threshold:
            filtered_annotations.append(ann)

    return filtered_annotations


if __name__ == "__main__":
    args = parse_args()
    image_dir_path = args.image_dir_path
    output_dir_path = args.output_dir_path
    sam_annotations_dir = args.sam_annotations_dir

    os.makedirs(output_dir_path, exist_ok=True)
    os.makedirs(f"{output_dir_path}/ov_sam", exist_ok=True)

    with open(args.image_names_txt_path, 'r') as f:
        image_names = [line.strip() for line in f if line.strip()]

    start_time = time.time()
    all_data = {}
    for image_name in tqdm(image_names):
        output_json_path = f"{output_dir_path}/ov_sam/{image_name[:-4]}.json"
        if os.path.exists(output_json_path):
            continue

        all_data[image_name] = {}
        image_path = f"{image_dir_path}/{image_name}"
        sam_json_path = f"{sam_annotations_dir}/{image_name[:-4]}.json"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        image = Image.open(image_path)
        orig_width, orig_height = image.size
        numpy_image, img_feat = extract_img_feat(image)
        numpy_height, numpy_width = numpy_image.shape[:2]

        with open(sam_json_path) as r:
            annotations = json.load(r)['annotations']

        all_image_predictions = []
        # Filter the annotations (remove all annotations with area less than 1% of the image size)
        annotations = filter_sam_boxes(annotations, 1)
        for i, ann in enumerate(annotations):
            bbox = ann['bbox']
            image, xyxy = get_bbox_with_draw(image, bbox, i)
            width_scale = numpy_width / orig_width
            height_scale = numpy_height / orig_height
            # Scale xyxy
            scaled_xyxy = [xyxy[0] * width_scale,  # x1
                           xyxy[1] * height_scale,  # y1
                           xyxy[2] * width_scale,  # x2
                           xyxy[3] * height_scale  # y2
                           ]
            bboxes = [[scaled_xyxy[0], scaled_xyxy[1]], [scaled_xyxy[2], scaled_xyxy[3]]]
            cls_info, class_name, score = run_inference(numpy_image, img_feat, bboxes)

            if score > 0.1:
                prediction = {}
                prediction['bbox'] = xyxy
                prediction['score'] = round(score, 2)
                prediction['label'] = class_name
                all_image_predictions.append(prediction)

        all_data[image_name]['ov_sam'] = [{k: json_serializable(v) for k, v in prediction.items()} for prediction in
                                          all_image_predictions]
        # Write all_data to a JSON file
        with open(output_json_path, 'w') as f:
            json.dump(all_data, f)
        all_data = {}

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- ov-same Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')
