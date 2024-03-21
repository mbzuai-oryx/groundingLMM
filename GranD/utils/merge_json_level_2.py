import argparse
import os
import numpy as np
import json
from tqdm import tqdm
from multiprocessing.pool import Pool

MODEL_NAMES = ['blip2', 'llava', 'mdetr-re']


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--predictions_dir_path", required=False, default="predictions")
    parser.add_argument("--output_dir_path", required=False, default="predictions/level-2-raw")

    parser.add_argument("--num_processes", required=False, type=int, default=32)

    args = parser.parse_args()

    return args


def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0


def worker(args, image_names):
    for image_name in tqdm(image_names):
        output_json_path = f"{args.output_dir_path}/{image_name}.json"
        if os.path.exists(output_json_path):
            continue
        image_name_with_extension = f"{image_name}.jpg"  # TODO: Handle other extensions
        all_data = {image_name_with_extension: {}}
        for model in MODEL_NAMES:
            json_file_path = f"{args.predictions_dir_path}/{model}/{image_name}.json"
            if is_file_empty(json_file_path):
                continue
            with open(json_file_path, 'r'):
                json_image_data = json.load(open(json_file_path, 'r'))
                if model in ['blip2', 'llava']:
                    if model == 'blip2':
                        json_image_data = json_image_data[image_name_with_extension][model]
                    else:
                        json_image_data = json_image_data[image_name_with_extension][model][model]
                    all_data[image_name_with_extension][model] = json_image_data
                elif "mdetr-re" in model:
                    json_image_data = json_image_data[image_name_with_extension]["mdetr-re"]
                    if "mdetr-re" in all_data[image_name_with_extension].keys():
                        all_data[image_name_with_extension]["mdetr-re"] = \
                            all_data[image_name_with_extension]["mdetr-re"] | json_image_data
                    else:
                        all_data[image_name_with_extension]["mdetr-re"] = json_image_data
                else:
                    pass

        # Save the level-1 json
        with open(output_json_path, 'w') as f:
            json.dump(all_data, f)


def split_list(input_list, n):
    """Split a list into 'n' parts using numpy."""
    arrays = np.array_split(np.array(input_list), n)
    return [arr.tolist() for arr in arrays]


def main():
    args = parse_args()  # Parse the arguments
    os.makedirs(args.output_dir_path, exist_ok=True)  # Create the directory to save the outputs

    # Get all image names and prepare the task-args
    all_image_names = os.listdir(f"{args.predictions_dir_path}/{MODEL_NAMES[-1]}")
    all_image_names = [image_name[:-5] for image_name in all_image_names]
    all_tasks_image_names_list = split_list(all_image_names, n=args.num_processes)
    task_args = [(args, task_image_names) for task_image_names in all_tasks_image_names_list]

    # Use a pool of workers to process the files in parallel.
    with Pool() as pool:
        pool.starmap(worker, task_args)


if __name__ == "__main__":
    main()
