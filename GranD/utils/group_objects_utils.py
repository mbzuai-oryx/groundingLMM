import numpy as np
import mmcv
import spacy
import re
import torch
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, CLIPSegForImageSegmentation
import torch.nn.functional as F

nlp = spacy.load('en_core_web_sm')

person_dict = {
    "woman": "person", "women": "persons",
    "man": "person", "men": "persons",
    "child": "person", "children": "persons",
    "boy": "person", "boys": "persons",
    "girl": "person", "girls": "persons",
    "teenager": "person", "teenagers": "persons",
    "adult": "person", "adults": "persons",
    "senior": "person", "seniors": "persons",
    "male": "person", "females": "persons",
    "infant": "person", "infants": "persons",
    "toddler": "person", "toddlers": "persons",
    "youth": "person", "elders": "persons",
    "adolescent": "person", "adolescents": "persons",
    "baby": "person", "babies": "persons",
    "elderly": "person", "teen": "person",
    "kid": "person", "kids": "persons",
    "gentleman": "person", "gentlemen": "persons",
    "lady": "person", "ladies": "persons"
}


class CLIPClassifier():
    def __init__(self, device):
        self.device = device
        self.model_init()

    def model_init(self):
        self.init_clip()
        # self.init_clipseg()

    def init_clip(self):
        # model_name = "openai/clip-vit-large-patch14"
        model_name = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)

    def init_clipseg(self):
        model_name = "CIDAS/clipseg-rd64-refined"
        self.clipseg_processor = AutoProcessor.from_pretrained(model_name)
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(self.device)
        self.clipseg_processor.image_processor.do_resize = False

    def clip_classification(self, image_path, bbox, class_list, top_k):
        img = mmcv.imread(image_path)
        box_patch = mmcv.imcrop(img, np.array([bbox[0], bbox[1], bbox[2], bbox[3]]), scale=1.4)
        inputs = self.clip_processor(text=class_list, images=box_patch, return_tensors="pt", padding=True).to(
            self.device)
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        if top_k == 1:
            return class_list[probs.argmax().item()]
        else:
            top_k_indices = probs.topk(top_k, dim=1).indices[0]
            return [class_list[index] for index in top_k_indices]

    def clipseg_segmentation(self, image_path, bbox, mask, class_list, scale_large=1.4):
        img = mmcv.imread(image_path)
        box_patch = mmcv.imcrop(img, np.array([bbox[0], bbox[1], bbox[2], bbox[3]]), scale=scale_large)

        inputs = self.clipseg_processor(
            text=class_list, images=[box_patch] * len(class_list),
            padding=True, return_tensors="pt").to(self.device)

        h, w = inputs['pixel_values'].shape[-2:]
        fixed_scale = (512, 512)
        inputs['pixel_values'] = F.interpolate(
            inputs['pixel_values'],
            size=fixed_scale,
            mode='bilinear',
            align_corners=False)

        outputs = self.clipseg_model(**inputs)
        logits = F.interpolate(outputs.logits[None], size=(h, w), mode='bilinear', align_corners=False)[0]
        class_ids_patch = logits.argmax(0)

        valid_mask = torch.tensor(mask).bool().squeeze(0)
        valid_mask_patch = mmcv.imcrop(valid_mask.numpy(),
                                       np.array([bbox[0], bbox[1], bbox[2], bbox[3]]), scale=scale_large)
        top_1_patch = torch.bincount(class_ids_patch[torch.tensor(valid_mask_patch)].flatten()).topk(1).indices
        top_1_mask_category = class_list[top_1_patch.item()]

        return top_1_mask_category


def get_noun_phrases(text):
    doc = nlp(text)
    nouns = []
    for chunk in doc.noun_chunks:
        if len(re.findall(r'\w+', chunk.text)) == 1:
            nouns.append(chunk.text)
    return nouns


def nms_same_model(boxes, labels, threshold=0.9):
    """Performs non-maximum suppression on the bounding boxes of the same model"""
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap for boxes with the same label
        overlap = np.zeros(len(idxs) - 1)
        for j in range(last):
            if labels[idxs[j]] == labels[i]:
                overlap[j] = (w[j] * h[j]) / area[idxs[j]]

        # Delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > threshold)[0])))

    return pick


def IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def IoU_mask(mask1, mask2):
    """
    Computes IoU based on segmentation masks.
    mask1, mask2: binary segmentation masks of the same size.
    """
    # Compute intersection and union
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    iou = np.sum(intersection) / np.sum(union)

    return iou


def dice_score_bbox(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the dice score by taking the intersection area and dividing it by the sum of areas of both bounding boxes
    dice_score = (2. * interArea) / float(boxAArea + boxBArea)

    return dice_score


def dice_score_mask(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)

    # compute the dice score by taking twice the intersection area and dividing it by the sum of total pixels in both masks
    dice_score = (2. * np.sum(intersection)) / (np.sum(mask1) + np.sum(mask2))

    return dice_score


def json_serializable(data):
    if data is None:
        return None
    if isinstance(data, np.float32):  # if it's a np.float32
        return round(float(data), 2)  # convert to python float
    elif isinstance(data, np.ndarray):  # if it's a np.ndarray
        return data.tolist()  # convert to python list
    elif isinstance(data, set):
        return list(data)
    else:  # for other types, let it handle normally
        return data
