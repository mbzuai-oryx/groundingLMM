import argparse
import os.path
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from utils import *
from tqdm import tqdm
from ddp import *
from torch.utils.data._utils.collate import default_collate

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser(description="MDETR Referring Expression")

    parser.add_argument("--image_dir_path", required=True)
    parser.add_argument("--output_dir_path", required=True)

    parser.add_argument("--blip2_pred_path", required=False, default="predictions/blip2")
    parser.add_argument("--llava_pred_path", required=False, default="predictions/llava")

    parser.add_argument("--ckpt_path", required=False, default="None",
                        help="Specify the checkpoints path if you want to load from a local path.")

    # DDP related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


def json_serializable(data):
    if isinstance(data, np.float32):  # if it's a np.float32
        return round(float(data), 2)  # convert to python float
    elif isinstance(data, np.ndarray):  # if it's a np.ndarray
        return data.tolist()  # convert to python list
    else:  # for other types, let it handle normally
        return data


def run_inference(model, image, image_size, nouns, phrases, threshold=0.5):
    num_phrases = len(phrases)
    image_b = image.repeat(num_phrases, 1, 1, 1)
    # propagate through the model
    memory_cache = model(image_b, phrases, encode_and_save=True)
    outputs = model(image_b, phrases, encode_and_save=False, memory_cache=memory_cache)

    all_phrase_boxes = {}
    all_nouns = []
    for i in range(num_phrases):
        probas = 1 - outputs['pred_logits'].softmax(-1)[i, :, -1].cpu()
        keep = (probas > threshold).cpu()
        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[i, keep], image_size)
        bboxes_scaled = bboxes_scaled.cpu().numpy().tolist()
        if bboxes_scaled:
            all_phrase_boxes[phrases[i]] = bboxes_scaled[0]
            all_nouns.append(nouns[i])

    return image, all_nouns, all_phrase_boxes


def run(args, dataloader, model_name="mdetr-re"):
    os.makedirs(f"{args.output_dir_path}/{model_name}", exist_ok=True)  # Create the output directory
    # Create the model and load checkpoints
    model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB3_refcocog',
                                          pretrained=True, return_postprocessor=True)

    ## Uncomment the following line if you want to load checkpoints from a local path
    # checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    # model.load_state_dict(checkpoint['model'], strict=False)

    model = model.cuda()
    model.eval()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    all_data = {}
    for (image_name, image, image_size, captions, nouns, phrases) in tqdm(dataloader):
        image_name, image_size, captions, nouns, phrases = (
            image_name[0], image_size[0], captions[0], nouns[0], phrases[0])

        output_file_name = f"{args.output_dir_path}/{model_name}/{image_name[:-4]}.json"
        if os.path.exists(output_file_name):
            continue
        try:
            _, all_nouns, all_phrase_boxes = run_inference(model, image, image_size, nouns, phrases)
        except Exception as e:
            print(f"Error processing image: {image_name}.")
            continue

        all_data[image_name] = {}
        all_data[image_name][model_name] = {}
        all_image_predictions = []
        for j, phrase in enumerate(all_phrase_boxes.keys()):
            noun = all_nouns[j]
            prediction = {}
            prediction['bbox'] = [round(float(b), 2) for b in all_phrase_boxes[phrase]]
            prediction['label'] = noun
            prediction['phrase'] = phrase
            all_image_predictions.append(prediction)

        all_data[image_name][model_name] = \
            [{k: json_serializable(v) for k, v in prediction.items()} for prediction in all_image_predictions]

        # Write all_data to a JSON file
        with open(output_file_name, 'w') as f:
            json.dump(all_data, f)
        all_data = {}


def custom_collate_fn(batch):
    image_names = [item[0] for item in batch]
    images = default_collate([item[1] for item in batch])
    image_sizes = [item[2] for item in batch]
    captions = [item[3] for item in batch]
    nouns = [item[4] for item in batch]
    phrases = [item[5] for item in batch]

    return image_names, images, image_sizes, captions, nouns, phrases


def main():
    args = parse_args()
    init_distributed_mode(args)

    # Create output directory if not exists
    os.makedirs(args.output_dir_path, exist_ok=True)

    # Create dataset
    image_dataset = CustomImageDataset(args.image_dir_path,
                                       args.blip2_pred_path, args.llava_pred_path, transform)
    distributed_sampler = DistributedSampler(image_dataset, rank=args.rank, shuffle=False)
    image_dataloader = DataLoader(image_dataset, batch_size=args.batch_size_per_gpu, num_workers=4,
                                  sampler=distributed_sampler, collate_fn=custom_collate_fn)

    run(args, image_dataloader)

    # Close LMDBs when exiting
    image_dataset.close()


if __name__ == "__main__":
    main()
