from group_objects_utils import nms_same_model, IoU, dice_score_bbox
import json
import os
import numpy as np
import argparse
import nltk
from nltk.corpus import words
from tqdm import tqdm
import re
from difflib import SequenceMatcher
from multiprocessing import Pool, cpu_count
import time

nltk.download("words")
word_set = set(words.words())

all_grounding_models = ['mdetr-re']
all_caption_models = ['blip2', 'llava']


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--raw_dir_path", required=False,
                        default="predictions/level-2-raw")
    parser.add_argument("--level_2_output_dir_path", required=False,
                        default="predictions/level-2-processed")

    parser.add_argument("--level_1_dir_path", required=False,
                        default="/predictions/level-1-processed")

    parser.add_argument("--num_processes", required=False, type=int, default=32)

    args = parser.parse_args()

    return args


def combine_models(image_predictions, nms_threshold=0.9, box_threshold_iou=0.65, mask_threshold_iou=0.65,
                   box_threshold_dice=0.75, mask_threshold_dice=0.75):
    """Function to group bounding boxes from different phrase grounding models based 5 factors:
        on their Intersection over Union (IoU)(4) and depth"""

    combined_data = {}
    model_data = {}
    for model_key in all_grounding_models:
        model_data[model_key] = {"random caption": image_predictions[model_key]}

        # Perform NMS within each model
        for key in model_data[model_key]:
            model_data[model_key][key] = [model_data[model_key][key][i] for i in nms_same_model(
                np.array([bbox['bbox'] for bbox in model_data[model_key][key]]),
                np.array([bbox['label'] for bbox in model_data[model_key][key]]), threshold=nms_threshold)]

    # Compare and combine bounding boxes from all models
    for key in model_data[all_grounding_models[0]]:
        # Convert label to a list if it's not already
        for bbox in model_data[all_grounding_models[0]][key]:
            bbox['label'] = [bbox['label'].strip()] if isinstance(bbox['label'], str) else [label.strip() for label in
                                                                                            bbox['label']]
        combined_data[key] = model_data[all_grounding_models[0]][
            key]  # add bboxes from the first model (also adds score and mask)
        for model_key in all_grounding_models[1:]:  # start from the second model
            if key in model_data[model_key]:  # comparing only with the same keys
                for bbox in model_data[model_key][key]:
                    matched = False
                    for combined_bbox in combined_data[key]:
                        if IoU(np.array(bbox['bbox']),
                               np.array(combined_bbox['bbox'])) > box_threshold_iou and dice_score_bbox(
                            np.array(bbox['bbox']), np.array(combined_bbox['bbox'])) > box_threshold_dice:
                            # combine bounding box entries
                            bbox['label'] = [bbox['label'].strip()] if isinstance(bbox['label'], str) else [
                                label.strip() for label in bbox['label']]
                            # combining labels and maintaining votes
                            combined_bbox['label'] = [combined_bbox['label']] if isinstance(combined_bbox['label'],
                                                                                            str) else combined_bbox[
                                'label']
                            combined_bbox['label'] = combined_bbox['label'] + bbox['label']
                            matched = True
                            break
                    if not matched:
                        combined_data[key].append(bbox)
    return combined_data


def normalize_phrase(phrase):
    return ' '.join(phrase.lower().split())


def tokenize(text):
    # Remove punctuation and split into tokens
    return re.findall(r'\b\w+\b', text)


def match_to_caption(label_tokens, caption_tokens):
    # Try to match each label to a section of the caption
    for i in range(len(caption_tokens)):
        if caption_tokens[i:i + len(label_tokens)] == label_tokens:
            return ' '.join(caption_tokens[i:i + len(label_tokens)])
    # If no match is found, return None
    return None


def process_phrase(phrase):
    words = phrase.split()
    result = []
    current_word = ""

    for word in words:
        current_word += word
        if current_word in word_set:
            result.append(current_word)
            current_word = ""

    if current_word:
        result.append(current_word)

    return " ".join(result)


def get_best_phrase(phrases, caption):
    # If all phrases are same, return the first one
    phrases = [process_phrase(p) for p in phrases]
    if all(x == phrases[0] for x in phrases):
        return phrases[0]
    else:
        # Tokenize the caption
        caption_tokens = tokenize(caption)

        # If the phrases are different, try to match each to the caption
        for phrase in phrases:
            # Tokenize the phrase and match to caption
            phrase_tokens = tokenize(phrase)
            match = match_to_caption(phrase_tokens, caption_tokens)

            # If a match was found, return it
            if match is not None:
                return match

        # If no matches were found, return the phrase that is most similar to the caption
        return max(phrases, key=lambda phrase: SequenceMatcher(None, phrase, caption).ratio())


def get_relationships(combined_data, level1_contents, match_threshold=0.75):
    """Function to determine relationships between objects identified by phrase grounding models and objects/floating objects
       from Level-1 content predictions"""

    # Initialize ID counter
    if 'id_counter' not in level1_contents:
        level1_contents['id_counter'] = len(level1_contents['objects']) + len(level1_contents['floating_objects']) + 1

    # for caption, _ in combined_data.items():
    #     # Initialize object_ids and floating_object_ids for each caption
    relationships = {'object_ids': [], 'floating_object_ids': [], 'grounding': []}

    for combined_bbox in combined_data:
        # Initialize match flags for objects and floating_objects
        object_matched = False
        floating_object_matched = False
        # Phrase grounding initialization
        grounding_entry = None

        # Normalize the label
        phrase = combined_bbox['phrase']
        normalized_label = normalize_phrase(phrase)

        # Try matching with objects and floating objects
        for object_type in ['objects', 'floating_objects']:
            for obj in level1_contents[object_type]:
                if IoU(np.array(combined_bbox['bbox']), np.array(obj['bbox'])) > match_threshold:

                    # Check if the phrase is already present in the grounding
                    for grounding in relationships['grounding']:
                        if grounding['phrase'] == normalized_label:
                            grounding_entry = grounding
                            break

                    if object_type == 'objects':
                        relationships['object_ids'].append(obj['id'])
                        object_matched = True
                        if grounding_entry is None:
                            grounding_entry = {'phrase': normalized_label, 'object_ids': [obj['id']]}
                            relationships['grounding'].append(grounding_entry)
                        else:
                            grounding_entry['object_ids'].append(obj['id'])

                    elif object_type == 'floating_objects':
                        relationships['floating_object_ids'].append(obj['id'])
                        floating_object_matched = True
                        if grounding_entry is None:
                            grounding_entry = {'phrase': normalized_label, 'object_ids': [obj['id']]}
                            relationships['grounding'].append(grounding_entry)
                        else:
                            grounding_entry['object_ids'].append(obj['id'])

                        # add label from combined_data to level1_contents floating_objects
                        # combining labels and maintaining votes
                        obj['labels'] = [obj['labels']] if isinstance(obj['labels'], str) else obj['labels']
                        combined_bbox['label'] = [combined_bbox['label']] if isinstance(combined_bbox['label'],
                                                                                        str) else combined_bbox[
                            'label']
                        obj['labels'] = obj['labels'] + combined_bbox['label']
                    break

        # If no match found with either objects or floating_objects, add to floating_objects
        if not object_matched and not floating_object_matched:
            # Create new floating object entry # TODO: Replace normalized label with label
            new_floating_object = {'bbox': combined_bbox['bbox'], 'labels': [normalized_label], 'attributes': None,
                                   'id': level1_contents['id_counter']  # Set ID as next available number
                                   }
            level1_contents['id_counter'] += 1
            # Add the new floating object to the list in level1_contents
            level1_contents['floating_objects'].append(new_floating_object)
            # Add the new floating object's ID to the floating_object_ids
            relationships['floating_object_ids'].append(new_floating_object['id'])
            # Check if the phrase is already present in the grounding
            for grounding in relationships['grounding']:
                if grounding['phrase'] == normalized_label:
                    grounding_entry = grounding
                    break
            # If not present, create a new grounding
            if grounding_entry is None:
                grounding_entry = {'phrase': normalized_label, 'object_ids': [new_floating_object['id']]}
                relationships['grounding'].append(grounding_entry)
            else:
                grounding_entry['object_ids'].append(new_floating_object['id'])

    return relationships, level1_contents


def update_level1_contents(level1_contents, relationship):
    """Update the level1_contents by moving floating_objects with more than one label to objects.
       Adjust object_ids accordingly in both level1_contents and relationship.
    """

    # Identify floating_objects with more than one label in the labels field
    to_be_moved = [obj for obj in level1_contents['floating_objects'] if len(obj['labels']) > 1]

    # Map of old IDs to new IDs
    id_map = {}

    # Move the selected floating_objects to objects and create the mapping
    for obj in to_be_moved:
        old_id = obj['id']
        level1_contents['objects'].append(obj)
        level1_contents['floating_objects'].remove(obj)
        new_id = len(level1_contents['objects']) - 1
        obj['id'] = new_id
        id_map[old_id] = new_id

    # Adjust the IDs for the remaining floating_objects
    for i, float_obj in enumerate(level1_contents['floating_objects'], start=len(level1_contents['objects'])):
        old_id = float_obj['id']
        float_obj['id'] = i
        id_map[old_id] = i

    # Update the relationship's object_ids using the map
    for grounding in relationship['grounding']:
        grounding['object_ids'] = [id_map.get(obj_id, obj_id) for obj_id in grounding['object_ids']]
    # Update object & floating object ids
    all_ids = relationship['object_ids'] + relationship['floating_object_ids']
    relationship['object_ids'], relationship['floating_object_ids'] = set(), set()
    floating_object_start_id = len(level1_contents['objects'])
    for obj_id in all_ids:
        new_obj_id = id_map.get(obj_id, obj_id)
        relationship['object_ids'].add(new_obj_id) if new_obj_id < floating_object_start_id else relationship[
            'floating_object_ids'].add(new_obj_id)
    relationship['object_ids'], relationship['floating_object_ids'] = list(relationship['object_ids']), list(
        relationship['floating_object_ids'])

    return level1_contents, relationship


def process_image(raw_file_path):
    # try:
    with open(raw_file_path, 'r') as f:
        merged_contents = json.load(f)
    # Collecting raw predictions from all models
    raw_item = {}
    level1_contents = {}
    for image in merged_contents.keys():
        raw_item[image] = {}
        for model in merged_contents[image].keys():
            raw_item[image][model] = merged_contents[image][model]
        # Get corresponding level-1 content
        image_name = image.rstrip('.jpg')
        level1_file_name = f'{image_name}.json'
        level1_file_path = os.path.join(level_1_dir_path, level1_file_name)
        with open(level1_file_path, 'r') as f:
            level1_contents[image] = json.load(f)

    image = list(raw_item.keys())[0]
    image_predictions = raw_item[image]
    level1_pred = level1_contents[image][image[:-4]]
    updated_level1 = {}
    # First combined predictions of the different phrase grounding models
    if len(all_grounding_models) > 1:
        combined_data = combine_models(image_predictions)
    else:
        combined_data = image_predictions
    relationship, level1_content = get_relationships(combined_data['mdetr-re'], level1_pred)
    level1_content_updated, relationship = update_level1_contents(level1_content, relationship)
    level1_content_updated['relationships'] = relationship

    updated_level1[image] = level1_content_updated

    # Add blip2 & llava captions
    updated_level1[image]['captions'] = [combined_data['blip2'], combined_data['llava']]

    image_name = image.rstrip('.jpg')
    lvl_2_output_file_path = os.path.join(level_2_output_dir_path, f'{image_name}.json')
    with open(lvl_2_output_file_path, 'w') as f:
        json.dump(updated_level1, f)

    return image, updated_level1
    # except Exception as e:
    #     print(f"Exception processing {raw_file_path}, {e}")
    #     return "abc", {}


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    raw_dir_path = args.raw_dir_path
    level_1_dir_path = args.level_1_dir_path
    level_2_output_dir_path = args.level_2_output_dir_path
    os.makedirs(level_2_output_dir_path, exist_ok=True)

    print(f'Loading raw files.')
    raw_files = os.listdir(raw_dir_path)
    raw_file_paths = []
    for i, file in enumerate(tqdm(raw_files)):
        image_name = file
        processed_json_path = os.path.join(args.level_2_output_dir_path, image_name)
        if not os.path.exists(processed_json_path):
            raw_file_paths.append(os.path.join(raw_dir_path, file))


    with Pool(args.num_processes) as pool:
        processed_images = dict(tqdm(pool.imap(process_image, raw_file_paths), total=len(raw_file_paths)))

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- level-2 Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')
