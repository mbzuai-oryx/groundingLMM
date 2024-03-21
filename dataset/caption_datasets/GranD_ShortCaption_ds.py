import os
import cv2
import lmdb
import json
import random
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from dataset.utils.utils import CAPTION_QUESTIONS
from tools.utils import DEFAULT_IMAGE_TOKEN


class GrandShortCaptionDataset(torch.utils.data.Dataset):
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
        self.base_dir = os.path.join(dataset_dir, "GranD_Data")
        self.image_folder = os.path.join(self.base_dir, "images")
        ann_file_name = "Grand_Caption_Grounding_lmdb"
        ann_path = os.path.join(self.base_dir, ann_file_name)
        self.annos = lmdb.open(ann_path, readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        mode = "Val" if validation else "Train"
        self.data_infos = self._load_annotations(os.path.join(self.base_dir, ann_file_name, f'{ann_file_name}_{mode}.txt'))
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        print('\033[92m' + "----CAP-{}: Grand Short Caption dataset initialized----".format(mode) + '\033[0m')

    def _load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            data_infos = [line.strip() for line in f if line.strip()]
        data_infos = data_infos[0: 1000] if self.validation else data_infos
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, labels):
        conversations = []
        questions = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []

        question = random.choice(CAPTION_QUESTIONS).strip()
        answer = labels

        conv.append_message(conv.roles[0], self.begin_str + question)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()
        conversations.append(prompt)
        return questions, conversations

    def __getitem__(self, idx):
        image_name = self.data_infos[idx] if (self.validation or not self.random_sampling) else self.data_infos[
            random.randint(0, len(self.data_infos) - 1)]
        # Get the annotation from lmdb
        with self.annos.begin() as txn:
            json_contents = txn.get(image_name.encode())
        json_contents = json.loads(json_contents.decode('utf-8'))
        ann_info = random.choice(json_contents[image_name])
        # Process the image
        image_path = os.path.join(self.image_folder, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Prepare input for Global Image Encoder
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        # Skip input for Grounding Image Encoder
        grounding_enc_image = None
        image_resize = None
        bboxes = None

        caption = ann_info["caption"]
        questions, conversations = self.create_conversations(caption)
        selected_labels = conversations

        masks = torch.rand(0, *image_resize)
        label = None

        assert len(conversations) == 1

        return (image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, image_resize,
                questions, selected_labels)