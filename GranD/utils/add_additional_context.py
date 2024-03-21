import argparse
import json
import os
from multiprocessing.pool import Pool
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--annotations_dir_path", required=False,
                        default="predictions/level-3-processed_with_masks")
    parser.add_argument("--level_4_additional_context_path", required=False,
                        default="predictions/level-4-vicuna_13B")
    parser.add_argument("--output_dir_path", required=False,
                        default="predictions/level-4-processed")

    parser.add_argument("--num_processes", required=False, type=int, default=32)

    args = parser.parse_args()

    return args


def process(args, task_files):
    for task_file in tqdm(task_files):
        annotation_path = os.path.join(args.annotations_dir_path, f"{task_file}.json")
        level_4_additional_context_path = os.path.join(args.level_4_additional_context_path, f"{task_file}.txt")
        output_path = os.path.join(args.output_dir_path, f"{task_file}.json")
        image_name = f"{task_file}.jpg"

        with open(annotation_path, 'r') as f:
            json_contents = json.load(f)

        with open(level_4_additional_context_path, 'r') as f:
            additional_context = f.read().strip()

        json_contents[image_name]["additional_context"] = additional_context

        with open(output_path, 'w') as f:
            json.dump(json_contents, f, indent=4)


def split_list(input_list, n):
    """Split a list into 'n' parts using numpy."""
    arrays = np.array_split(np.array(input_list), n)
    return [arr.tolist() for arr in arrays]


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir_path, exist_ok=True)

    all_files = os.listdir(args.annotations_dir_path)
    all_files = [file[:-5] for file in all_files]

    all_tasks_files = split_list(all_files, n=args.num_processes)
    task_args = [(args, task_file) for task_file in all_tasks_files]

    # Use a pool of workers to process the files in parallel.
    with Pool() as pool:
        pool.starmap(process, task_args)
