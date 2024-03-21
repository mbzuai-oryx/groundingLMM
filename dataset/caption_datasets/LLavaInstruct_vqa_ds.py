import os
import cv2
import json
import random
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN


class LLaVAInstructDataset(torch.utils.data.Dataset):
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=10000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=True):

        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample
        self.validation = validation
        self.random_sampling = random_sampling

        # Defining paths
        mode = "val" if validation else "train"
        self.base_dir = os.path.join(dataset_dir, "llava_dataset")
        self.image_folder = os.path.join(dataset_dir, f"coco_2017/{mode}2017")
        annotations_file = os.path.join(self.base_dir, "llava_instruct_150k.json")
        self.data_infos = self._load_annotations(annotations_file)
        print('\033[92m' + "----CAP-{}: LLaVA-Instruct VQA dataset initialized----".format(mode) + '\033[0m')

    def _load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            data_infos = json.load(f)
        data_infos = data_infos[0: 1000] if self.validation else data_infos
        return data_infos

    def __len__(self):
        return len(self.vqa_data)

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, conv_ann):
        # Preprocess:
        for sentence in conv_ann:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = (sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip())
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                    )
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        if roles[conv_ann[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            conv_ann = conv_ann[1:]

        for j, sentence in enumerate(conv_ann):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{j}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
        questions = conversations

        return questions, conversations

    def __getitem__(self, idx):
        ann_info = self.data_infos[idx] if (self.validation or not self.random_sampling) else self.data_infos[
            random.randint(0, len(self.data_infos) - 1)]
        image_path = os.path.join(self.image_folder, ann_info["image"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Prepare input for Global Image Encoder
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        # Skip input for Grounding Image Encoder
        grounding_enc_image = None
        image_resize = None
        bboxes = None

        conv_ann = ann_info["conversations"]
        questions, conversations = self.create_conversations(conv_ann)
        selected_labels = conversations

        masks = None
        label = None

        assert len(conversations) == 1

        return (image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, image_resize,
                questions, selected_labels)