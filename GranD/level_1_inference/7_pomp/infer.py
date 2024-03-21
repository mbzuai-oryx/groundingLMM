import argparse
import torch
import os
import json
import numpy as np
import cv2
from utils import TextEncoder
from tqdm import tqdm
import clip
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# Detic libraries
from third_party.CenterNet2.centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detectron2.utils.visualizer import _create_text_labels
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--image_names_txt_path", required=True)
    parser.add_argument("--image_dir_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--tags_dir_path", required=True)

    args = parser.parse_args()

    return args


def run_pomp_detector(text_encoder, model, ctx, predictor, image_path, classes,
                      device='cuda', output_score_threshold=0.05):
    im = cv2.imread(image_path)

    n_ctx = ctx.size(0)
    classnames = [name.replace("_", " ") for name in classes]
    prompt_prefix = " ".join(["X"] * n_ctx)
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).int().to(device)

    with torch.no_grad():
        embedding = model.token_embedding(tokenized_prompts).type(model.dtype)
        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + n_ctx:, :]
        ctx = ctx.unsqueeze(0).expand(len(classes), -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        text_features = text_encoder(prompts, tokenized_prompts)

    classifier = text_features.t().float()
    num_classes = len(classes)
    reset_cls_test(predictor.model, classifier, num_classes)
    # Reset visualization threshold
    for cascade_stages in range(len(predictor.model.roi_heads.box_predictor)):
        predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold

    # Run model and show results
    outputs = predictor(im)

    return outputs["instances"]


def json_serializable(data):
    if isinstance(data, np.float32):  # if it's a np.float32
        return round(float(data), 2)  # convert to python float
    elif isinstance(data, np.ndarray):  # if it's a np.ndarray
        return data.tolist()  # convert to python list
    else:  # for other types, let it handle normally
        return data


def is_json_file_empty(filename):
    # Check if the file itself has no content
    if os.path.getsize(filename) == 0:
        return True

    # Check if the parsed content is empty
    with open(filename, 'r') as file:
        content = json.load(file)
        return not bool(content)


def main():
    args = parse_args()

    image_dir_path = args.image_dir_path
    output_dir_path = args.output_dir_path
    tags_dir_path = args.tags_dir_path

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_name = "pomp"

    processed_image_dir = os.path.join(output_dir_path, model_name)
    os.makedirs(processed_image_dir, exist_ok=True)
    processed_images = os.listdir(processed_image_dir)

    # Build the detic model
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file("configs/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size_pomp_cross_datasets.yaml")
    cfg.MODEL.WEIGHTS = ('Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size_pomp.pth')

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True  # For better visualization purpose. Set to False for all classes.
    # cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
    predictor = DefaultPredictor(cfg)

    # Build the CLIP model with POMP
    pomp_ckpts = f"vit_b16_ep20_randaug2_unc1000_16shots_nctx16_cscFalse_ctpend_seed42.pth.tar"

    ckpt = torch.load(pomp_ckpts, map_location='cpu')
    ctx = ckpt["state_dict"]['ctx'].to(device)
    # Loading CLIP
    model, preprocess = clip.load('ViT-B/16', device=device)
    text_encoder = TextEncoder(model)

    start_time = time.time()
    all_data = {}
    with open(args.image_names_txt_path, 'r') as f:
        image_names = [line.strip() for line in f if line.strip()]

    for image_name in tqdm(image_names):
        output_file_name = f'{image_name[:-4]}.json'
        if output_file_name in processed_images:
            continue
        image_path = f"{image_dir_path}/{image_name}"

        try:
            tag2text_tags = json.load(open(f"{tags_dir_path}/tag2text/{image_name[:-4]}.json", 'r'))
        except Exception as e:
            print(f"Tag2Text exception at image {image_name}")
            tag2text_tags = {image_name: {"tag2text": {"tags": []}}}
        try:
            ram_tags = json.load(open(f"{tags_dir_path}/ram/{image_name[:-4]}.json", 'r'))
        except Exception as e:
            print(f"Ram exception at image {image_name}")
            ram_tags = {image_name: {"ram": {"tags": []}}}

        tag2text_tags = tag2text_tags[image_name]["tag2text"]["tags"]
        ram_tags = ram_tags[image_name]["ram"]["tags"]
        classes = list(set(tag2text_tags + ram_tags))
        classes.append('trees')
        if len(classes) == 0:
            print(f"Skipping image {image_name} as no tags are found.")
            continue

        all_data[image_name] = {}

        predictions = run_pomp_detector(text_encoder, model, ctx, predictor, image_path, classes)

        bboxes = predictions.pred_boxes.tensor if predictions.has("pred_boxes") else None
        scores = predictions.scores.cpu().numpy().tolist() if predictions.has("scores") else None
        labels = _create_text_labels(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None,
                                     None, classes)

        all_image_predictions = []

        # Check if xyxy is not empty
        if len(bboxes) > 0:
            for j, box in enumerate(bboxes.cpu().numpy().tolist()):
                prediction = {}
                prediction['bbox'] = [round(float(b), 2) for b in box]
                # prediction['sam_mask'] = masks[j].cpu().numpy()
                prediction['score'] = round(scores[j], 2)
                prediction['label'] = labels[j]
                all_image_predictions.append(prediction)

            all_data[image_name]['pomp'] = [{k: json_serializable(v) for k, v in prediction.items()} for
                                            prediction in all_image_predictions]

        with open(os.path.join(processed_image_dir, output_file_name), 'w') as f:
            json.dump(all_data, f)
        all_data = {}

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "----POMP Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')


if __name__ == "__main__":
    main()
