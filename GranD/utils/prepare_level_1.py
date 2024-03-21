import json
import os
import argparse
from group_objects_utils import *
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from affordances.add_affordance import convert_label_to_category, get_affordances
from affordances.category_descriptions import categories as affordance_categories
import warnings
import numpy as np
import time
from PIL import Image


def numpy_warning_filter(message, category, filename, lineno, file=None, line=None):
    return 'numpy' in filename


warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--image_dir_path", required=False, default="images")
    parser.add_argument("--output_dir_path", required=False, default="predictions/level-1-processed")
    parser.add_argument("--raw_dir_path", required=False, default="predictions/level-1-raw")

    args = parser.parse_args()

    return args


all_model_keys = ['eva-02-01', 'co_detr', 'eva-02-02', 'ov_sam', 'owl_vit', 'pomp', 'grit']
thresholds = {"eva-02-02": 0.6, "co_detr": 0.2, "eva-02-01": 0.6, "ov_sam": 0.3, "owl_vit": 0.1, "pomp": 0.3,
              "grit": 0.3}
single_person_threshold = {"eva-02-02": 0.75, "co_detr": 0.5, "eva-02-01": 0.75, "owl_vit": 0.2, "pomp": 0.6,
                           "grit": 0.6}


def compute_all_iou_and_dice(bboxes):
    num_boxes = len(bboxes)
    iou_matrix = np.zeros((num_boxes, num_boxes))
    dice_matrix = np.zeros((num_boxes, num_boxes))

    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            iou_val = IoU(bboxes[i]['bbox'], bboxes[j]['bbox'])
            dice_val = dice_score_bbox(bboxes[i]['bbox'], bboxes[j]['bbox'])
            iou_matrix[i, j], iou_matrix[j, i] = iou_val, iou_val
            dice_matrix[i, j], dice_matrix[j, i] = dice_val, dice_val

    return iou_matrix, dice_matrix


def bbox_size(bbox):
    """Calculate the area of a bounding box."""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def add_big_bboxes(image, objects, floating_objects):
    # get image size
    imag_path = os.path.join(image_dir_path, image)
    img = Image.open(imag_path)
    image_width, image_height = img.size
    half_image_size = 0.5 * image_height * image_width

    # Compute the biggest bounding box size across objects and floating_objects
    biggest_bbox_size = max([bbox_size(obj['bbox']) for obj in objects + floating_objects], default=0)

    # Move floating objects to objects if conditions are met
    for float_obj in floating_objects.copy():  # Using .copy() to prevent modifying list during iteration
        if bbox_size(float_obj['bbox']) > half_image_size and bbox_size(float_obj['bbox']) > biggest_bbox_size:
            objects.append(float_obj)
            floating_objects.remove(float_obj)

    return objects, floating_objects


def group_bounding_boxes(models_data: dict, affordance_label_to_category, image):
    """Function to group bounding boxes from different models based on their Intersection over Union (IoU)"""
    global person_threshold
    checked_bboxes = set()  # Stores checked bounding boxes
    objects = []
    floating_objects = []
    floating_attributes = []
    box_threshold_iou = 0.65
    box_threshold_dice = 0.75

    for i, model_name in enumerate(all_model_keys):
        if model_name in models_data:
            for j in range(len(models_data[model_name])):
                if (model_name, j) in checked_bboxes:  # Skip this bbox if already checked
                    continue

                bbox_i = models_data[model_name][j]['bbox']
                score_i = models_data[model_name][j]['score']
                label_i = models_data[model_name][j]['label']

                # Flag to check if a match was found
                is_matched = False
                new_object = {'bbox': bbox_i, 'labels': set(), 'score': score_i, 'attributes': set(), }

                if model_name != 'grit':  # For all detectors
                    new_object['labels'].add(label_i)
                else:  # For GRiT Model
                    new_object['attributes'].add(label_i)

                for k, other_model_name in enumerate(all_model_keys[i + 1:]):  # Start from next model
                    # Avoid checking previous bboxes of the same model
                    start_l = j + 1 if other_model_name == model_name else 0
                    if other_model_name in models_data:
                        for l in range(start_l, len(models_data[other_model_name])):
                            if (other_model_name, l) in checked_bboxes:  # Skip this bbox if already checked
                                continue

                            bbox_k = models_data[other_model_name][l]['bbox']
                            label_k = models_data[other_model_name][l]['label']

                            # If there's enough overlap of bboxes
                            if (IoU(bbox_i, bbox_k) >= box_threshold_iou and dice_score_bbox(bbox_i,
                                                                                             bbox_k) >= box_threshold_dice):

                                is_matched = True

                                if other_model_name != 'grit':  # For all detectors
                                    new_object['labels'].add(label_k)
                                else:  # For GRiT Model
                                    new_object['attributes'].add(label_k)

                                checked_bboxes.add((other_model_name, l))  # Mark bbox_k as checked

                if is_matched:
                    objects.append(new_object)
                elif model_name != 'grit':
                    # 3 scenarios
                    person_threshold = single_person_threshold.get(model_name, 0)
                    original_threshold = thresholds.get(model_name, 0)
                    # Special case for the 'person' label to add as object without voting
                    if model_name in ['eva-02-01', 'co_detr',
                                      'eva-02-02'] and label_i == 'person' and score_i > person_threshold:
                        objects.append(new_object)
                    # Special case for the 'person' label - setting threshold back to original to add ad floating
                    elif model_name in ['eva-02-01', 'co_detr',
                                        'eva-02-02'] and label_i == 'person' and score_i > original_threshold:
                        floating_objects.append(new_object)
                    elif label_i != 'person':
                        floating_objects.append(new_object)
                else:
                    floating_attributes.append(new_object)

                if len(new_object['attributes']) == 0:
                    new_object['attributes'] = None
                checked_bboxes.add((model_name, j))  # Mark bbox_i as checked

    # Assigning object ids to objects and floating objects
    object_id = 0
    for obj in objects:
        obj['id'] = object_id
        object_id += 1

    for obj in floating_objects:
        obj['id'] = object_id
        object_id += 1

    # Adding large boxes from floating to valid objects
    # objects, floating_objects = add_big_bboxes(image, objects, floating_objects)

    objects = [{k: json_serializable(v) for k, v in prediction.items()} for prediction in objects]
    floating_objects = [{k: json_serializable(v) for k, v in prediction.items()} for prediction in floating_objects]
    floating_attributes = [{k: json_serializable(v) for k, v in prediction.items()} for prediction in
                           floating_attributes]

    processed_level_1_dict = {'objects': objects, 'floating_objects': floating_objects,
                              'floating_attributes': floating_attributes,
                              'landmark': models_data['landmark'] if 'landmark' in models_data.keys() else ""}
    processed_level_1_dict = get_affordances(processed_level_1_dict, affordance_label_to_category)

    return processed_level_1_dict


def process_image(raw_file_path):
    with open(raw_file_path, 'r') as f:
        merged_contents = json.load(f)

    # Collecting raw predictions from all models based on thresholds
    image = list(merged_contents.keys())[0]
    all_bbox_dict = {}
    all_bbox_dict[image] = {}
    for model in merged_contents[image].keys():
        if model in all_model_keys:
            model_data = [process_prediction(pred, model) for pred in merged_contents[image][model] if
                          process_prediction(pred, model)]
        else:
            model_data = merged_contents[image][model]
        all_bbox_dict[image][model] = model_data

    # all_bbox_dict[image]['landmark'] = merged_contents[image]['landmark']

    # image_predictions = merged_contents[image]  # TODO: Fix this BIG bug
    image_predictions = all_bbox_dict[image]  # TODO: This is how it is fixed
    # depth_threshold = 400

    # Initialize affordance
    affordance_label_to_category = convert_label_to_category(affordance_categories)
    processed_level_1_dict = group_bounding_boxes(image_predictions, affordance_label_to_category, image)

    # Replace all man, women, men, woman with person
    for pred in processed_level_1_dict['objects']:
        labels = pred['labels']
        pred['labels'] = ['person' if label.lower() in ['man', 'woman', 'men', 'women'] else label for label in labels]

    image_name = image.rstrip('.jpg')
    output_file_path = os.path.join(f"{output_dir_path}", f'{image_name}.json')
    with open(output_file_path, 'w') as f:
        json.dump({image_name: processed_level_1_dict}, f)

    return image, processed_level_1_dict


def get_threshold_for_prediction(model, label):
    # Special case for the 'person' label
    if model in ['eva-02-01', 'co_detr', 'eva-02-02'] and label == 'person':
        return 0.2
    return thresholds.get(model, 0)


def process_prediction(prediction, model):
    threshold = get_threshold_for_prediction(model, prediction.get('description', prediction.get('label')))
    if prediction['score'] > threshold:
        label = prediction.get('description', prediction.get('label'))
        return {'bbox': prediction['bbox'], 'label': label, 'score': prediction['score']}
    return None


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    image_dir_path = args.image_dir_path
    raw_dir_path = args.raw_dir_path
    output_dir_path = args.output_dir_path
    depth_map_dir = args.depth_map_dir

    os.makedirs(output_dir_path, exist_ok=True)

    print(f'Loading raw predictions of {all_model_keys}')
    raw_files = os.listdir(raw_dir_path)
    raw_file_paths = []
    for file in raw_files:
        image_name = file.replace("_level_1_raw.json", "")
        processed_json_path = os.path.join(output_dir_path, f'{image_name}_level_1_processed.json')
        if not os.path.exists(processed_json_path):
            raw_file_paths.append(os.path.join(raw_dir_path, file))

    print(f'Processing raw predictions of {all_model_keys}')
    with Pool(cpu_count()) as pool:
        processed_image = dict(tqdm(pool.imap(process_image, raw_file_paths), total=len(raw_file_paths)))
    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- level-1 Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')
