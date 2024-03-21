import json
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset preparation script")

    parser.add_argument("--data_dir_path", required=True,
                        help="Specify the directory containing GranD annotation JSON files")
    parser.add_argument("--output_dir_path", required=True,
                        help="Define the directory where processed annotation JSONs will be saved.")
    args = parser.parse_args()

    return args


def correct_attribute(attribute):
    if ":" in attribute:
        attribute = ' '.join(attribute.split(':')[1:])
        attribute = attribute.strip()

    return attribute


def remove_underscores(labels):
    return [label.replace('_', ' ') for label in labels]


def create_annotation(raw_file_path):
    try:
        with open(raw_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        image = list(data.keys())[0]
        ann_dict = {}
        ann_dict[image] = []
        json_contents = data[image]
        # Start prep of ann
        objects = json_contents['objects']
        for obj in objects:
            attributes = obj.get('attributes')
            labels = obj.get('labels')
            labels = remove_underscores(labels)
            if attributes:
                if isinstance(attributes, str):
                    attributes = [attributes]
            else:
                continue

            filt_attributes = []
            for attr in attributes:
                # Remove prefixes with different forms using regex
                attr = re.sub(r'(?i)\bAssistant:?\s*', '', attr)  # Removes 'Assistant' prefix in a case-insensitive manner
                attr = re.sub(r'(?i)\bistant:?\s*', '', attr)  # Removes 'istant' prefix in a case-insensitive manner
                filt_attributes.append(attr)

            if len(filt_attributes) == 1:
                re_dict = {'attribute': filt_attributes[0], 'id': obj['id'], 'bbox': obj['bbox'], 'segmentation': obj['segmentation']}
                ann_dict[image].append(re_dict)
            elif len(filt_attributes) > 1:
                attr_added = False  # Flag to check if any attribute has been added
                for attr in filt_attributes:
                    # Check if any of the labels are present in the attribute
                    if any(label in attr for label in labels):
                        re_dict = {'attribute': attr, 'id': obj['id'], 'bbox': obj['bbox'], 'segmentation': obj['segmentation']}
                        ann_dict[image].append(re_dict)
                        attr_added = True
                if not attr_added:
                    re_dict = {'attribute': filt_attributes[0], 'id': obj['id'], 'bbox': obj['bbox'], 'segmentation': obj['segmentation']}
                    ann_dict[image].append(re_dict)

        image_name = image.rstrip('.jpg')
        output_file_path = os.path.join(f"{output_dir_path}", f'{image_name}.json')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(ann_dict, f, ensure_ascii=False, indent=2)

        return image
    except Exception as e:
        print(f"Error processing file {raw_file_path}: {e}")
        return None


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

    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(create_annotation, raw_file_paths), total=len(raw_file_paths)):
            pass
    # if Debug:
    # for f in tqdm(raw_file_paths, desc="Processing", unit="file"):
    #     processed_image = create_annotation(f)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- Referring-expression taken: {} seconds ----".format(elapsed_time) + '\033[0m')