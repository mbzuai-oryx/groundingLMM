import os
import cv2
import lmdb
import json
import numpy as np
import random
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import REGION_QUESTIONS


class GrandReferRegDataset(torch.utils.data.Dataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, max_gt_per_img=10, validation=False, random_sampling=True):
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)
        self.max_gt_per_img = max_gt_per_img
        self.validation = validation
        self.random_sampling = random_sampling

        # Defining paths
        self.base_dir = os.path.join(dataset_dir, "GranD_Data")
        self.image_folder = os.path.join(self.base_dir, "images")
        ann_file_name = "Grand_Referring_Expression_lmdb"
        ann_path = os.path.join(self.base_dir, ann_file_name)
        self.annos = lmdb.open(ann_path, readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        mode = "Val" if validation else "Train"
        self.data_infos = self._load_annotations(
            os.path.join(self.base_dir, ann_file_name, f'{ann_file_name}_{mode}.txt')
            )
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        self.question_templates = REGION_QUESTIONS
        print('\033[92m' + "----REGION-{}: GranD Referring Region dataset initialized----".format(mode) + '\033[0m')

    def _load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            data_infos = [line.strip() for line in f if line.strip()]
        data_infos = data_infos[0: 1000] if self.validation else data_infos
        return data_infos

    def _parse_annotations(self, ann_info):
        annotations = {'bboxes': [], 'labels': []}
        for ann in ann_info:
            bbox = ann['bbox']
            if bbox:
                annotations['bboxes'].append(bbox)
                annotations['labels'].append(ann['attribute'])

        annotations['bboxes'] = np.array(annotations['bboxes'], dtype=np.float32) if annotations[
            'bboxes'] else np.zeros((0, 4), dtype=np.float32)
        return annotations

    def __getitem__(self, index):
        image_name = self.data_infos[index] if (self.validation or not self.random_sampling) \
            else self.data_infos[random.randint(0, len(self.data_infos) - 1)]
        image_path = os.path.join(self.image_folder, image_name)
        # Get the annotation from lmdb
        with self.annos.begin() as txn:
            json_contents = txn.get(image_name.encode())
        json_contents = json.loads(json_contents.decode('utf-8'))
        ann_info = json_contents[image_name]
        ann = self._parse_annotations(ann_info)

        data_item = {
            "image_path": image_path,
            "filename": image_name,
            "bbox": ann['bboxes'],
            "labels": ann['labels'],
        }

        return self.process_data(data_item)

    def __len__(self):
        return len(self.coco.imgs)

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def region_enc_processor(self, orig_size, post_size, bboxes, labels, device):
        orig_h, orig_w = orig_size
        post_h, post_w = post_size
        y_scale = post_h / orig_h
        x_scale = post_w / orig_w
        shuffle_ids = torch.randperm(len(labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids_reg_question = shuffle_ids[:self.max_gt_per_img]
            selected_labels = [labels[i] for i in shuffle_ids_reg_question]
        else:
            selected_labels = [labels[i] for i in shuffle_ids]
        selected_bboxes = bboxes[shuffle_ids]
        # Ensure selected_bboxes is two-dimensional
        if len(selected_bboxes.shape) == 1:
            selected_bboxes = np.expand_dims(selected_bboxes, axis=0)

        selected_bboxes[:, [0, 2]] *= x_scale
        selected_bboxes[:, [1, 3]] *= y_scale
        selected_bboxes = torch.tensor(selected_bboxes, device=device, dtype=torch.float32) / post_h
        return selected_bboxes, selected_labels

    def create_conversations(self, labels, question_templates):
        questions = []
        answers = []
        for i, label in enumerate(labels):
            question = random.choice(question_templates).strip().replace('<region>', f'region{i + 1} <bbox>')
            questions.append(question)
            answers.append(label)

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                question = self.begin_str + question
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())
        return questions, conversations

    def process_data(self, data_item):
        data_labels = data_item['labels']
        data_bboxes = data_item['bbox']

        image_path = data_item['image_path']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        # Prepare input for Global Image Encoder
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        post_h, post_w = global_enc_image.shape[1:3]
        # Skip input for Grounding Image Encoder
        grounding_enc_image = None
        image_resize = None
        # Prepare input for Region Image Encoder
        bboxes, selected_labels = self.region_enc_processor(
            (orig_h, orig_w), (post_h, post_w), data_bboxes, data_labels, global_enc_image.device
            )
        masks = None

        questions, conversations = self.create_conversations(selected_labels, question_templates=self.question_templates)
        label = None

        return (image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, image_resize,
                questions, selected_labels)
