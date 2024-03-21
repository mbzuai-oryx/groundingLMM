import torch
from tqdm import tqdm
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from midas.model_loader import default_models, load_model
from ddp import *
import utils
import time

first_execution = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir_path", required=True,
                        help="Path to the directory containing images.")
    parser.add_argument("--output_dir_path", required=True,
                        help="Path to the output directory to store the predictions.")

    parser.add_argument('-m', '--model_weights', default=None,
                        help='Path to the trained weights of model')
    parser.add_argument('-t', '--model_type', default='dpt_beit_large_512', help='Model type')
    parser.add_argument('-s', '--side', action='store_true', default=True,
                        help='Output images contain RGB and depth images side by side')

    parser.add_argument("--opts", required=False, default="")

    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    return args


def process(model, sample, target_size):
    """
    Run the inference and interpolate.

    Args:
        model: the model used for inference
        image: the image fed into the neural network
        target_size: the size (width, height) the neural network output is interpolated to

    Returns:
        the prediction
    """
    global first_execution

    if first_execution:
        height, width = sample.shape[2:]
        print(f"    Input resized to {width}x{height} before entering the encoder")
        first_execution = False

    prediction = model.forward(sample)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )

    return prediction


def run(output_path, model_path, model_type="dpt_beit_large_512", dataloader=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize=False, height=None,
                                                square=False)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # Create output folder
    start_time = time.time()
    model_name = 'midas'
    processed_image_dir = os.path.join(args.output_dir_path, model_name)
    os.makedirs(processed_image_dir, exist_ok=True)
    processed_images = os.listdir(processed_image_dir)
    for (image_name, image, image_size) in tqdm(dataloader):
        image_name = image_name[0]
        output_file_name = image_name[:-4]
        if output_file_name in processed_images:
            continue
        # Compute
        with torch.no_grad():
            prediction = process(model, image, image_size)

        # Output
        if output_path is not None:
            filename = os.path.join(processed_image_dir, output_file_name)
            # utils.write_pfm(filename + ".pfm", prediction.astype(np.float32))
            utils.write_jpeg(filename, prediction.astype(np.float32))

    print("Finished")
    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- Midas Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')


if __name__ == "__main__":
    args = parse_args()

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # Set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    init_distributed_mode(args)
    image_dir_path = args.image_dir_path

    # Set up output paths
    output_dir_path = args.output_dir_path
    Path(output_dir_path).mkdir(parents=True, exist_ok=True)

    batch_size_per_gpu = args.batch_size_per_gpu

    # Create dataset
    image_dataset = CustomImageDataset(image_dir_path)
    distributed_sampler = DistributedSampler(image_dataset, rank=args.rank, shuffle=False)
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size_per_gpu, num_workers=4,
                                  sampler=distributed_sampler)

    # Compute depth maps
    run(output_dir_path, args.model_weights, args.model_type, dataloader=image_dataloader)
