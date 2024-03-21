from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple
import json
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import relu, sigmoid
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import numpy as np
import time
from ram.models import ram, tag2text
from ddp import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--image_dir_path", required=True,
                        help="Path to the directory containing images.")
    parser.add_argument("--output_dir_path", required=True,
                        help="Path to the output directory to store the predictions.")

    # model
    parser.add_argument("--model-type", type=str, choices=("ram", "tag2text"), required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--backbone", type=str, choices=("swin_l", "swin_b"), default=None,
                        help="If `None`, will judge from `--model-type`")
    parser.add_argument("--open-set", action="store_true",
                        help=("Treat all categories in the taglist file as unseen and perform open-set classification. "
                              "Only works with RAM."))
    # data
    parser.add_argument("--input-size", type=int, default=384)
    # threshold
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--threshold", type=float, default=None,
                       help=("Use custom threshold for all classes. Mutually xclusive with `--threshold-file`. If both "
                             "`--threshold` and `--threshold-file` is `None`, will use a default threshold setting."))
    # miscellaneous
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--opts", required=False, default="")

    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    # post process and validity check
    args.model_type = args.model_type.lower()

    assert not (args.model_type == "tag2text" and args.open_set)
    if args.backbone is None:
        args.backbone = "swin_l" if args.model_type == "ram" else "swin_b"

    return args


def get_lag_list(model_name):
    if model_name == 'ram':
        tag_path = 'ram/data/ram_tag_list.txt'
    else:
        tag_path = 'ram/data/tag_list.txt'
    with open(tag_path, "r", encoding="utf-8") as f:
        taglist = [line.strip() for line in f]

    return taglist


def get_class_idxs(taglist: List[str]) -> Optional[List[int]]:
    """Get indices of required categories in the label system."""
    class_index = [taglist.index(tag) for tag in taglist]
    return class_index


def load_thresholds(threshold: Optional[float], model_type: str, class_idxs: List[int], num_classes: int, ) -> List[
    float]:
    """Decide what threshold(s) to use."""
    if not threshold:  # use default
        if model_type == "ram":
            # use class-wise tuned thresholds
            ram_threshold_file = "ram/data/ram_tag_list_threshold.txt"
            with open(ram_threshold_file, "r", encoding="utf-8") as f:
                idx2thre = {idx: float(line.strip()) for idx, line in enumerate(f)}
                return [idx2thre[idx] for idx in class_idxs]
        else:
            return [0.65] * num_classes
    else:
        return [threshold] * num_classes


def gen_pred_file(imglist: List[str], tags: List[List[str]], img_root: str, pred_file: str) -> None:
    """Generate text file of tag prediction results."""
    with open(pred_file, "w", encoding="utf-8") as f:
        for image, tag in zip(imglist, tags):
            # should be relative to img_root to match the gt file.
            s = str(Path(image).relative_to(img_root))
            if tag:
                s = s + "," + ",".join(tag)
            f.write(s + "\n")


def load_ram(backbone: str, checkpoint: str, input_size: int, class_idxs: List[int], ) -> Module:
    model = ram(pretrained=checkpoint, image_size=input_size, vit=backbone)
    model.label_embed = Parameter(model.label_embed[class_idxs, :])
    return model.to(device).eval()


def load_tag2text(backbone: str, checkpoint: str, input_size: int) -> Module:
    model = tag2text(pretrained=checkpoint, image_size=input_size, vit=backbone)
    return model.to(device).eval()


@torch.no_grad()
def forward_ram(model: Module, imgs: Tensor) -> Tensor:
    image_embeds = model.module.image_proj(model.module.visual_encoder(imgs.to(device)))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
    label_embed = relu(model.module.wordvec_proj(model.module.label_embed)).unsqueeze(0).repeat(imgs.shape[0], 1, 1)
    tagging_embed, _ = model.module.tagging_head(encoder_embeds=label_embed, encoder_hidden_states=image_embeds,
                                                 encoder_attention_mask=image_atts, return_dict=False, mode='tagging', )
    return sigmoid(model.module.fc(tagging_embed).squeeze(-1))


@torch.no_grad()
def forward_tag2text(model: Module, class_idxs: List[int], imgs: Tensor) -> Tensor:
    image_embeds = model.module.visual_encoder(imgs.to(device))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
    label_embed = model.module.label_embed.weight.unsqueeze(0).repeat(imgs.shape[0], 1, 1)
    tagging_embed, _ = model.module.tagging_head(encoder_embeds=label_embed, encoder_hidden_states=image_embeds,
                                                 encoder_attention_mask=image_atts, return_dict=False, mode='tagging', )
    return sigmoid(model.module.fc(tagging_embed))[:, class_idxs]


def json_serializable(data):
    if isinstance(data, np.float32):  # if it's a np.float32
        return round(float(data), 2)  # convert to python float
    elif isinstance(data, np.ndarray):  # if it's a np.ndarray
        return data.tolist()  # convert to python list
    else:  # for other types, let it handle normally
        return data


def run_inference(args, dataloader):
    # Get tag list, class idx, and set threshold
    taglist = get_lag_list(args.model_type)
    class_idxs = get_class_idxs(taglist)
    thresholds = load_thresholds(threshold=args.threshold, model_type=args.model_type,
                                 class_idxs=class_idxs, num_classes=len(taglist))

    # Load model
    if args.model_type == "ram":
        model = load_ram(backbone=args.backbone, checkpoint=args.checkpoint, input_size=args.input_size,
                         class_idxs=class_idxs)
    else:
        model = load_tag2text(backbone=args.backbone, checkpoint=args.checkpoint, input_size=args.input_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    start_time = time.time()
    all_data = {}
    model_name = args.model_type

    processed_image_dir = os.path.join(args.output_dir_path, model_name)
    os.makedirs(processed_image_dir, exist_ok=True)

    processed_images = os.listdir(processed_image_dir)
    for (image_name, image) in tqdm(dataloader):
        image_name = image_name[0]
        output_file_name = f'{image_name[:-4]}.json'
        if output_file_name in processed_images:
            continue
        if args.model_type == "ram":
            out = forward_ram(model, image)
        else:
            out = forward_tag2text(model, class_idxs, image)
        logits = out.cpu()

        pred_tags = []
        pred_scores = []
        for i, s in enumerate(logits.squeeze(0)):
            if s >= thresholds[i]:
                pred_tags.append(taglist[i])
                pred_scores.append(round(s.item(), 2))

        all_image_predictions = {'tags': pred_tags, 'scores': pred_scores}

        all_data[image_name] = {}
        all_data[image_name][model_name] = {}
        all_data[image_name][model_name] = all_image_predictions

        # Write all_data to a JSON file (file_wise)
        with open(os.path.join(processed_image_dir, output_file_name), 'w') as f:
            json.dump(all_data, f)
        all_data = {}

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- Tags Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')


if __name__ == "__main__":
    args = parse_args()
    init_distributed_mode(args)
    image_dir_path = args.image_dir_path

    batch_size_per_gpu = args.batch_size_per_gpu

    # Create dataset
    image_dataset = CustomImageDataset(image_dir_path, args.input_size)
    distributed_sampler = DistributedSampler(image_dataset, rank=args.rank, shuffle=False)
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size_per_gpu, num_workers=4,
                                  sampler=distributed_sampler)

    run_inference(args, dataloader=image_dataloader)
