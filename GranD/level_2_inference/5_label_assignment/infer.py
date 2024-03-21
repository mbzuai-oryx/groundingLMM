import argparse
import time
from tqdm import tqdm
from eva_clip import create_model_and_transforms, get_tokenizer
from torch.utils.data import DataLoader, DistributedSampler
from ddp import *
import torch

MODEL_NAME = "EVA02-CLIP-L-14-336"
PRETRAINED = "eva02_clip_224to336"


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--image_dir_path", required=True)
    parser.add_argument("--level_2_dir_path", required=False,
                        default="predictions/level-2-processed_gpt4roi")
    parser.add_argument("--output_dir_path", required=False,
                        default="predictions/level-2-processed_eva_clip")
    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


def run_inference(model, tokenizer, box_crops, labels):
    """
    model: Clip model
    box_crops: {id1: crop1, id2: crop2, id3: crop3, ...}
    labels: {id1: [labels1], id2: [labels2], id3: [labels3], ...}
    """
    assigned_labels = {}
    all_box_crops = []
    all_labels = []
    label_id_to_indexes = {}

    for id in labels.keys():
        box_labels = labels[id]
        all_box_crops.append(box_crops[id])
        start_idx = len(all_labels)
        all_labels += box_labels
        end_idx = len(all_labels)
        label_id_to_indexes[id] = [start_idx, end_idx]

    all_box_tensor = torch.stack(all_box_crops)
    text = tokenizer(all_labels).to('cuda')
    with torch.no_grad(), torch.cuda.amp.autocast():
        all_box_features = model.module.encode_image(all_box_tensor.to('cuda'))
        all_label_features = model.module.encode_text(text)

        for i, id in enumerate(label_id_to_indexes.keys()):
            start_idx, end_idx = label_id_to_indexes[id]
            box_labels = all_labels[start_idx:end_idx]

            current_box_features = all_box_features[i]
            current_label_features = all_label_features[start_idx:end_idx]

            current_box_features /= current_box_features.norm(dim=-1, keepdim=True)
            current_label_features /= current_label_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * current_box_features @ current_label_features.T).softmax(dim=-1)
            label_idx = text_probs.argmax()
            label = box_labels[label_idx]

            assigned_labels[id] = label

    return assigned_labels


def custom_collate(batch):
    image_names, box_crops_list, labels_list = zip(*batch)

    box_crops_collated = {}
    labels_collated = {}

    for box_crops, labels in zip(box_crops_list, labels_list):
        box_crops_collated.update(box_crops)
        labels_collated.update(labels)

    return image_names, box_crops_collated, labels_collated


def update_json_with_labels(json_contents, id_to_label_dict):
    for obj_category in ['objects', 'floating_objects']:
        for obj in json_contents[obj_category]:
            obj_id = obj['id']
            # Use the provided dictionary to set the 'label' key for each object
            # Default to 'Unknown' if the id is not in the dictionary
            if obj_id in id_to_label_dict.keys():
                obj['label'] = id_to_label_dict.get(obj_id)
            else:
                obj['label'] = obj['labels'][0]

    return json_contents


def main():
    args = parse_args()
    init_distributed_mode(args)

    os.makedirs(args.output_dir_path, exist_ok=True)

    # Create model
    model, _, preprocess = create_model_and_transforms(MODEL_NAME, PRETRAINED, force_custom_clip=True)
    tokenizer = get_tokenizer(MODEL_NAME)
    model = model.to("cuda")
    model.eval()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # Create dataset
    image_dataset = CustomLevel1JsonDataset(args.image_dir_path, args.level_2_dir_path,
                                            args.output_dir_path, image_transform=preprocess)
    distributed_sampler = DistributedSampler(image_dataset, rank=args.rank, shuffle=True)
    image_dataloader = DataLoader(image_dataset, batch_size=args.batch_size_per_gpu,
                                  num_workers=1, sampler=distributed_sampler, collate_fn=custom_collate)
    start_time = time.time()
    for image_name, box_crops, labels in tqdm(image_dataloader):
        image_name = image_name[0]
        output_json_file_path = f"{args.output_dir_path}/{image_name[:-4]}.json"
        if os.path.exists(output_json_file_path):
            continue
        if len(box_crops) == 0:
            continue
        assigned_labels = run_inference(model, tokenizer, box_crops, labels)

        with open(output_json_file_path, 'w') as f:
            json.dump(assigned_labels, f)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- label assignment Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')


if __name__ == "__main__":
    main()
