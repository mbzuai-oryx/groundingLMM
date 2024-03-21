import copy
from io import BytesIO
import requests
import torch
from PIL import Image
from gpt4roi.train.train import preprocess, preprocess_multimodal

DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'
MIN_AREA_RATIO = 0.02  # 2 %

multimodal_cfg = {'is_multimodal': True,
                  'sep_image_conv_front': False,
                  'image_token_len': 256,
                  'image_aspect_ratio': 'square',
                  'use_im_start_end': True}


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def get_init_inputs(img_path, processor, tokenizer):
    image = load_image(img_path)
    image_size = image.size
    image = processor.preprocess(image,
                                 do_center_crop=False,
                                 return_tensors='pt')['pixel_values'][0]

    image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                            size=(224, 224),
                                            mode='bilinear',
                                            align_corners=False).squeeze(0)

    cur_token_len = (image.shape[1] // 14) * (image.shape[2] // 14)  # FIXME: 14 is hardcoded patch size

    begin_str = """The <image> provides an overview of the picture.\n"""
    question_str = 'Give a short description of region1 <bbox>.'

    init_question = begin_str + question_str
    sources = dict()
    sources['conversations'] = []
    sources['conversations'].append(
        {'from': 'human', 'value': init_question})
    sources = preprocess_multimodal([sources['conversations']],
                                    multimodal_cfg, cur_token_len)
    ori_source = copy.deepcopy(sources)

    data_dict = preprocess(
        sources,
        tokenizer)

    data_dict = dict(input_ids=data_dict['input_ids'][0],
                     labels=data_dict['labels'][0],
                     sources=ori_source,
                     init_question=init_question,
                     )

    data_dict['image'] = image

    data_dict['img_metas'] = dict(filename=img_path)

    return data_dict, image_size


def get_bboxes(img_size, detections):
    w, h = img_size
    x1, y1, x2, y2 = detections
    # Normalize bboxes from predictions
    pred = [x1 / w, y1 / h, x2 / w, y2 / h]

    pred_bboxes = [pred]
    torch_pred_bboxes = torch.Tensor(pred_bboxes)

    return torch_pred_bboxes


def filter_objects_by_size(objects, image_size, min_area_ratio):
    """
    Filters bounding boxes based on their size relative to the image size.

    Args:
        objects (list of dicts): List of object dictionaries containing 'bbox' and other keys.
        image_size (tuple of ints): Image size as (width, height).
        min_area_ratio (float): The minimum area ratio threshold to keep a bounding box.

    Returns:
        list of dicts: List of filtered objects.
    """
    img_w, img_h = image_size
    img_area = img_w * img_h

    filtered_object_ids = [obj['id'] for obj in objects if (obj['bbox'][2] - obj['bbox'][0]) * (
            obj['bbox'][3] - obj['bbox'][1]) / img_area >= min_area_ratio]

    return filtered_object_ids


def add_big_bboxes(image_size, objects, floating_objects):
    try:
        # Get image size
        image_width, image_height = image_size
        half_image_size = 0.5 * image_height * image_width

        # Compute the biggest bounding box size across objects only
        if objects:
            biggest_bbox_size = max(bbox_size(obj['bbox']) for obj in objects)
        else:
            biggest_bbox_size = 0

        # Find the highest scoring floating object for each label
        highest_score_per_label = {}
        for float_obj in floating_objects:
            float_obj_bbox_size = bbox_size(float_obj['bbox'])
            if float_obj_bbox_size > half_image_size and float_obj_bbox_size > biggest_bbox_size:
                for label in float_obj['labels']:
                    if label not in highest_score_per_label or float_obj['score'] > highest_score_per_label[label]['score']:
                        highest_score_per_label[label] = float_obj

        # Add the highest scoring floating objects to objects list, avoiding duplicate labels
        existing_labels = {label for obj in objects for label in obj['labels']}
        for label, float_obj in highest_score_per_label.items():
            if label not in existing_labels or float_obj['score'] > max(
                    obj['score'] for obj in objects if label in obj['labels']):
                objects.append(float_obj)
                existing_labels.add(label)

        # Update floating_objects list to remove the objects that were moved to objects list
        floating_objects[:] = [obj for obj in floating_objects if obj not in highest_score_per_label.values()]
    except Exception as e:
        pass


def bbox_size(bbox):
    """Calculate the area of a bounding box."""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
