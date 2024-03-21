import os
import cv2
import random
import numpy as np
import torch
from pycocotools.coco import COCO
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import REGION_QUESTIONS


class RegionBaseDataset(torch.utils.data.Dataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, max_gt_per_img=10, validation=False, dataset_name='',
                 image_dir='', json_path='', intro_string='', question_templates=None, random_sampling=True):
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

        # Dataset type specific
        self.begin_str = intro_string
        self.base_dir = os.path.join(dataset_dir, dataset_name)
        self.ann_file = os.path.join(self.base_dir, json_path)
        self.question_templates = question_templates
        self.image_folder = os.path.join(self.base_dir, image_dir)

        self.data_infos = self._load_annotations(self.ann_file)
        self.data_infos = [self.data_infos[i] for i in self._filter_images(min_size=32)]

    def _load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        img_ids = self.coco.getImgIds()
        data_infos = []
        for img_id in img_ids:
            if self.validation and len(data_infos) == 1000:
                # limited images during validation
                break
            info = self.coco.loadImgs([img_id])[0]
            info['filename'] = info['file_name'].split('_')[-1]
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])
            data_infos.append(info)
        return data_infos

    def _filter_images(self, min_size):
        return [i for i, info in enumerate(self.data_infos) if min(info['width'], info['height']) >= min_size]

    def _parse_annotations(self, img_info, ann_info):
        annotations = {'bboxes': [], 'labels': [], 'bboxes_ignore': [], 'masks_ann': [],
                       'seg_map': img_info['file_name'].replace('jpg', 'png')}

        for ann in ann_info:
            if ann.get('ignore', False) or ann['area'] <= 0 or ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
                continue
            bbox = self._get_valid_bbox(ann['bbox'], img_info['width'], img_info['height'])
            if bbox:
                annotations['bboxes'].append(bbox)
                annotations['labels'].append(img_info['caption'].strip())

        annotations['bboxes'] = np.array(annotations['bboxes'], dtype=np.float32) if annotations[
            'bboxes'] else np.zeros((0, 4), dtype=np.float32)
        annotations['bboxes_ignore'] = np.zeros((0, 4), dtype=np.float32)
        return annotations

    def _get_valid_bbox(self, bbox, img_width, img_height):
        x1, y1, w, h = bbox
        inter_w = max(0, min(x1 + w, img_width) - max(x1, 0))
        inter_h = max(0, min(y1 + h, img_height) - max(y1, 0))
        if inter_w * inter_h == 0:
            return None
        return [x1, y1, x1 + w, y1 + h]

    def __getitem__(self, index):
        img_info = self.data_infos[index] if (self.validation or not self.random_sampling) \
            else self.data_infos[random.randint(0, len(self.data_infos) - 1)]
        ann_info = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_info['id']))
        ann = self._parse_annotations(img_info, ann_info)

        data_item = {
            "image_path": os.path.join(self.image_folder, img_info['file_name']),
            "width": img_info['width'],
            "height": img_info['height'],
            "bbox": ann['bboxes'],
            "caption": img_info['caption'],
            "labels": ann['labels'],
            "seg_map": ann['seg_map'],
        }

        return self.process_data(data_item)

    def __len__(self):
        return len(self.data_infos)

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
        shuffle_ids = torch.randperm(len(labels))[:self.max_gt_per_img]
        selected_bboxes = bboxes[shuffle_ids]

        # Ensure selected_bboxes is two-dimensional
        if len(selected_bboxes.shape) == 1:
            selected_bboxes = np.expand_dims(selected_bboxes, axis=0)

        selected_labels = [labels[i] for i in shuffle_ids]
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
        bboxes, selected_labels = self.region_enc_processor((orig_h, orig_w), (post_h, post_w), data_bboxes, data_labels,
                                                            global_enc_image.device)
        masks = None

        questions, conversations = self.create_conversations(
            selected_labels, question_templates=self.question_templates
            )
        label = None

        return (image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, image_resize,
                questions, selected_labels)


class RefCocoRegDataset(RegionBaseDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, max_gt_per_img=10, validation=False, random_sampling=True):
        intro_string = DEFAULT_IMAGE_TOKEN + "\n" + ("I will provide you with only one region containing only one "
                                                     "object, although there may be other objects present in the "
                                                     "image. It is recommended that you describe the object's "
                                                     "relative position with respect to other objects in the image, "
                                                     "as well as its position within the image and its basic "
                                                     "attributes.")
        json_path = os.path.join("mdetr_annotations", "finetune_refcoco_train.json")
        dataset_name = "RefCoco_Reg"
        image_dir = "coco_2014"
        question_templates = ['<region>',]
        mode = "Val" if validation else "Train"

        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, epoch_samples, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            intro_string, question_templates, random_sampling
            )
        print('\033[92m' + "----REGION-{}: Loaded RefCOCO dataset ----".format(mode) + '\033[0m')


class RefCocoGRegDataset(RegionBaseDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, max_gt_per_img=10, validation=False, random_sampling=True):
        intro_string = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        dataset_name = "RefCoco_Reg"
        json_files = {'validation': "finetune_refcocog_val.json", 'training': "finetune_refcocog_train.json"}
        json_path = os.path.join("mdetr_annotations", json_files['validation'] if validation else json_files['training'])
        image_dir = "coco_2014"
        question_templates = REGION_QUESTIONS
        mode = "Val" if validation else "Train"

        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, epoch_samples, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            intro_string, question_templates, random_sampling
            )
        print('\033[92m' + "----REGION-{}: Loaded RefCOCO-G dataset ----".format(mode) + '\033[0m')


class RefCocoPRegDataset(RegionBaseDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, max_gt_per_img=10, validation=False, random_sampling=True):
        intro_string = DEFAULT_IMAGE_TOKEN + "\n" + ("I will provide you with only one region containing only one "
                                                     "object, although there may be other objects present in the "
                                                     "image. It is recommended that you describe the object's "
                                                     "relative position with respect to other objects in the image, "
                                                     "as well as its position within the image and its basic "
                                                     "attributes.")
        dataset_name = "RefCoco_Reg"
        json_files = {'validation': "finetune_refcoco+_val.json", 'training': "finetune_refcoco+_train.json"}
        json_path = os.path.join(
            "mdetr_annotations", json_files['validation'] if validation else json_files['training']
            )
        image_dir = "coco_2014"
        question_templates = ['<region>', ]
        mode = "Val" if validation else "Train"

        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, epoch_samples, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            intro_string, question_templates, random_sampling
        )
        print('\033[92m' + "----REGION-{}: Loaded RefCOCO-P dataset ----".format(mode) + '\033[0m')


class VisualGenomeRegDataset(RegionBaseDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, max_gt_per_img=10, validation=False, random_sampling=True):
        intro_string = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        dataset_name = "visual_genome"
        json_files = {'validation': "test_caption.json", 'training': "train.json"}
        json_path = json_files['validation'] if validation else json_files['training']
        image_dir = "images"
        question_templates = REGION_QUESTIONS
        mode = "Val" if validation else "Train"

        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, epoch_samples, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            intro_string, question_templates, random_sampling
            )
        print('\033[92m' + "----REGION-{}: Loaded VisualGenome dataset ----".format(mode) + '\033[0m')

    def _parse_annotations(self, img_info, ann_info):
        annotations = {'bboxes': [], 'labels': [], }

        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            # Check for valid area and dimensions
            if ann['area'] <= 0 or ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
                continue
            bbox = self._get_valid_bbox(ann['bbox'], img_info['width'], img_info['height'])
            if bbox:
                annotations['bboxes'].append(bbox)
                annotations['labels'].append(ann['caption'].strip())

        annotations['bboxes'] = np.array(annotations['bboxes'], dtype=np.float32) if annotations[
            'bboxes'] else np.zeros((0, 4), dtype=np.float32)
        return annotations

    def __getitem__(self, index):
        img_info = self.data_infos[index] if (self.validation or not self.random_sampling) \
            else self.data_infos[random.randint(0, len(self.data_infos) - 1)]
        ann_info = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_info['id']))
        ann = self._parse_annotations(img_info, ann_info)

        data_item = {
            "image_path": os.path.join(self.image_folder, img_info['file_name']),
            "width": img_info['width'],
            "height": img_info['height'],
            "bbox": ann['bboxes'],
            "labels": ann['labels'],
        }

        return self.process_data(data_item)
