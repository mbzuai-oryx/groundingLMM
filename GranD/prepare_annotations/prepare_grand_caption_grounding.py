import json
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset preparation script")

    parser.add_argument("--data_dir_path", required=True,
                        help="Specify the directory containing GranD annotation JSON files")
    parser.add_argument("--output_dir_path", required=True,
                        help="Define the directory where processed annotation JSONs will be saved.")
    args = parser.parse_args()

    return args


def create_annotation(raw_file_path):
    try:
        with open(raw_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        image = list(data.keys())[0]
        ann_dict = {}
        ann_dict[image] = []
        json_contents = data[image]
        objects = json_contents['objects']
        floating_objects = json_contents['floating_objects']
        segmentation_floating_objects = False
        def get_segm_by_id(search_id):
            for obj in objects:
                if obj['id'] == search_id:
                    return obj['segmentation']
            # Search in floating object if they have segmentation
            if segmentation_floating_objects:
                for obj in floating_objects:
                    if obj['id'] == search_id:
                        return obj['segmentation']
            return None

        # Start prep of ann
        captions = json_contents['short_captions']
        for caption_dict in captions:
            # Add segmentation to each phrase
            for phrase in caption_dict["details"]:
                phrase['segmentation'] = get_segm_by_id(phrase["id"])
            ann_dict[image].append(caption_dict)

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
    # For Debug:
    # for f in raw_file_paths:
    #     processed_image = create_annotation(f)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- Caption-Grounding taken: {} seconds ----".format(elapsed_time) + '\033[0m')