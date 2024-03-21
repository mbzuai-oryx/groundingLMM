import json
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import time
import unicodedata
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Short Caption Grounding")

    parser.add_argument("--data_dir_path", required=False,
                        default="predictions/level-2-processed_labelled")
    parser.add_argument("--output_dir_path", required=False,
                        default="predictions/short_captions_grounded")

    parser.add_argument("--num_processes", required=False, type=int, default=32)

    args = parser.parse_args()

    return args


def traverse_and_correct(obj):
    if isinstance(obj, dict):
        return {k: traverse_and_correct(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [traverse_and_correct(ele) for ele in obj]
    elif isinstance(obj, str):
        # Replace double backslashes
        corrected_str = obj.replace('\\\\', '\\')
        # Normalize Unicode strings (optional, e.g., to NFC, NFD, NFKC, NFKD)
        normalized_str = unicodedata.normalize('NFKC', corrected_str)
        return normalized_str
    else:
        return obj


def create_annotation(args, file_paths):
    for raw_file_path in tqdm(file_paths):
        try:
            with open(raw_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data = traverse_and_correct(data)
            # Collecting raw predictions from all models based on thresholds
            image = list(data.keys())[0]
            ann_dict = {}
            ann_dict = []
            json_contents = data[image]
            # Start prep of ann
            captions = json_contents['captions']
            relationships = json_contents['relationships']
            objects = json_contents['objects']
            floating_objects = json_contents['floating_objects']

            def get_bbox_by_id(search_id):
                for obj in objects:
                    if obj['id'] == search_id:
                        return obj['bbox']
                for obj in floating_objects:
                    if obj['id'] == search_id:
                        return obj['bbox']
                return None

            for caption in captions:
                caption = caption.strip()
                caption_dict = {'caption': caption, 'details': []}

                # Check if a phrase from relationships exists in the caption
                for relation in relationships["grounding"]:
                    phrase = relation["phrase"]
                    if phrase in caption:
                        start = caption.find(phrase)
                        end = start + len(phrase)
                        bbox = get_bbox_by_id(relation["object_ids"][0])
                        smallest_id = min(relation["object_ids"])
                        detail = {
                            'phrase': phrase,
                            'tokens_positive': [start, end],
                            'id': smallest_id,
                            'bbox': bbox
                        }
                        caption_dict['details'].append(detail)

                ann_dict.append(caption_dict)

            data[image]['short_captions'] = ann_dict
            _ = data[image].pop('captions', None)

            image_name = image.rstrip('.jpg')
            output_file_path = os.path.join(f"{output_dir_path}", f'{image_name}.json')

            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error processing file {raw_file_path}: {e}")


def split_list(input_list, n):
    """Split a list into 'n' parts using numpy."""
    arrays = np.array_split(np.array(input_list), n)
    return [arr.tolist() for arr in arrays]


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    data_dir_path = args.data_dir_path
    output_dir_path = args.output_dir_path

    os.makedirs(output_dir_path, exist_ok=True)

    raw_files = os.listdir(data_dir_path)
    raw_file_paths = []
    for file in raw_files:
        processed_json_path = os.path.join(output_dir_path, f'{file}.json')
        if not os.path.exists(processed_json_path):
            raw_file_paths.append(os.path.join(data_dir_path, file))

    all_tasks_raw_file_names_list = split_list(raw_file_paths, n=args.num_processes)
    task_args = [(args, raw_file) for raw_file in all_tasks_raw_file_names_list]
    with Pool() as pool:
        pool.starmap(create_annotation, task_args)
