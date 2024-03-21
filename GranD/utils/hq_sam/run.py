import argparse
import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from ddp import CustomJsonDataset, init_distributed_mode
from rle_format import mask_to_rle_pytorch, coco_encode_rle
from segment_anything import sam_model_registry, SamPredictor
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--image_dir_path", required=True)
    parser.add_argument("--level_3_processed_path", required=False,
                        default="predictions/level-3-processed")
    parser.add_argument("--output_dir_path", required=False,
                        default="predictions/level-3-processed_with_masks")

    parser.add_argument("--sam_encoder_version", required=False, default="vit_h")
    parser.add_argument("--checkpoints_path", required=False, default="sam_hq_vit_h.pth")

    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


def custom_collate(batch):
    image_paths, image_names, json_contents, box_list = zip(*batch)

    box_collated = {}
    for box_crops in box_list:
        box_collated.update(box_crops)

    return image_paths, image_names, json_contents, box_collated


def update_json_with_masks(json_contents, image_name, id_to_mask_dict):
    for obj_category in ['objects', 'floating_objects']:
        for obj in json_contents[image_name][obj_category]:
            obj_id = obj['id']
            # Use the provided dictionary to set the 'label' key for each object
            # Default to 'Unknown' if the id is not in the dictionary
            if obj_id in id_to_mask_dict.keys():
                obj['segmentation'] = id_to_mask_dict.get(obj_id)
                obj['segmentation_source'] = "HQ_SAM"

    return json_contents


def post_process(args, image_name, sam_masks, object_ids, json_contents):
    uncompressed_mask_rles = mask_to_rle_pytorch(sam_masks)
    sam_masks_list = [coco_encode_rle(uncompressed_mask_rles[i]) for i in range(len(uncompressed_mask_rles))]

    mask_dict = dict(zip(object_ids, sam_masks_list))
    updated_json_contents = update_json_with_masks(json_contents, image_name, mask_dict)
    output_json_file_path = f"{args.output_dir_path}/{image_name[:-4]}.json"

    with open(output_json_file_path, 'w') as f:
        json.dump(updated_json_contents, f, indent=4)


def convert_xywh_to_xyxy(boxes_xywh):
    """
    Convert a list of bounding boxes from xywh format to xyxy format.

    :param boxes_xywh: List of boxes in xywh format
                       [(x1, y1, w1, h1), (x2, y2, w2, h2), ...]

    :return: List of boxes in xyxy format
             [(x1, y1, x1_w1, y1_h1), (x2, y2, x2_w2, y2_h2), ...]
    """
    boxes_xyxy = []
    for box in boxes_xywh:
        x, y, w, h = box
        boxes_xyxy.append((x, y, x + w, y + h))

    return boxes_xyxy


def main():
    args = parse_args()
    init_distributed_mode(args)

    os.makedirs(f"{args.output_dir_path}", exist_ok=True)

    # Load HQ-SAM model
    model_type = args.sam_encoder_version
    sam = sam_model_registry[model_type](checkpoint=args.checkpoints_path)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)

    # Create dataset
    dataset = CustomJsonDataset(args.image_dir_path, args.level_3_processed_path,
                                f"{args.output_dir_path}")
    distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu,
                            num_workers=1, sampler=distributed_sampler, collate_fn=custom_collate)

    for image_path, image_name, json_contents, bboxes in tqdm(dataloader):
        image_path, image_name, json_contents = image_path[0], image_name[0], json_contents[0]
        output_json_file_path = f"{args.output_dir_path}/{image_name[:-4]}.json"
        if not len(bboxes) == 0:
            if os.path.exists(output_json_file_path):
                continue

            object_ids = list(bboxes.keys())
            boxes = list(bboxes.values())

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            boxes_xyxy = convert_xywh_to_xyxy(boxes)
            input_box = torch.tensor(boxes_xyxy, device=predictor.device)
            transformed_box = predictor.transform.apply_boxes_torch(input_box, image.shape[:2])
            input_point, input_label = None, None
            batch_box = False if input_box is None else len(input_box) > 1

            hq_token_only = False
            masks, scores, logits = predictor.predict_torch(
                point_coords=input_point,
                point_labels=input_label,
                boxes=transformed_box,
                multimask_output=False,
                hq_token_only=hq_token_only,
            )

            post_process(args, image_name, masks.squeeze(1).to("cpu"), object_ids, json_contents)
        else:
            if json_contents:
                with open(output_json_file_path, 'w') as f:
                    json.dump(json_contents, f)


if __name__ == "__main__":
    main()
