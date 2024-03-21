import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import DataLoader, DistributedSampler
from ddp import *
from models.grit_src.image_dense_captions import setup_grit
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir_path", required=True)
    parser.add_argument("--output_dir_path", required=True)

    parser.add_argument("--opts", required=False, default="")
    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


def json_serializable(data):
    if isinstance(data, np.float32):  # if it's a np.float32
        return float(data)  # convert to python float
    elif isinstance(data, np.ndarray):  # if it's a np.ndarray
        return data.tolist()  # convert to python list
    else:  # for other types, let it handle normally
        return data


def dense_pred_dict(predictions):
    boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
    scores = predictions["instances"].scores if predictions["instances"].has("scores") else None
    object_description = predictions["instances"].pred_object_descriptions.data

    prediction_list = []
    for i in range(len(object_description)):
        bbox = [round(float(a), 2) for a in boxes[i].tensor.cpu().detach().numpy()[0]]
        score = round(float(scores[i]), 2) if scores is not None else None

        prediction_dict = {
            'bbox': bbox,
            'score': score,
            'description': object_description[i],
        }
        prediction_list.append(prediction_dict)

    return prediction_list


def run_inference(args, dataloader):
    # Initialize model
    dense_caption_demo = setup_grit(device=device)
    dense_caption_model = dense_caption_demo.predictor.model
    model = torch.nn.parallel.DistributedDataParallel(dense_caption_model, device_ids=[args.gpu])
    start_time = time.time()
    all_data = {}
    model_name = 'grit'
    processed_image_dir = os.path.join(args.output_dir_path, model_name)
    os.makedirs(processed_image_dir, exist_ok=True)
    processed_images = os.listdir(processed_image_dir)
    for (image_name, image, height, width) in tqdm(dataloader):
        image_name = image_name[0]
        output_file_name = f'{image_name[:-4]}.json'
        if output_file_name in processed_images:
            continue
        inputs = {"image": image[0], "height": int(height), "width": int(width)}
        with torch.no_grad():
            predictions = model([inputs])[0]
            all_data[image_name] = {}
            all_data[image_name][model_name] = dense_pred_dict(predictions)

        # Write all_data to a JSON file (file_wise)
        with open(os.path.join(processed_image_dir, output_file_name), 'w') as f:
            json.dump(all_data, f)
        all_data = {}
    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- Grit Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')


if __name__ == "__main__":
    args = parse_args()
    init_distributed_mode(args)
    image_dir_path = args.image_dir_path

    # set up output paths
    output_dir_path = args.output_dir_path
    Path(output_dir_path).mkdir(parents=True, exist_ok=True)

    batch_size_per_gpu = args.batch_size_per_gpu

    # Create dataset
    image_dataset = CustomImageDataset(image_dir_path)
    distributed_sampler = DistributedSampler(image_dataset, rank=args.rank, shuffle=False)
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size_per_gpu, num_workers=4,
                                  sampler=distributed_sampler)

    run_inference(args, dataloader=image_dataloader)
