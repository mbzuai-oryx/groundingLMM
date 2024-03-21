import argparse
import os
import json
import re
import numpy as np
from tqdm import tqdm
import spacy
from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import string

punctuation_chars = string.punctuation
punctuation_chars = punctuation_chars.replace(",", "").replace(".", "")

spacy.prefer_gpu()
# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")


def parse_args():
    parser = argparse.ArgumentParser(description="Dense Caption Grounding")

    parser.add_argument("--level_3_dense_caption_txt_dir_path", required=False,
                        default="predictions/level-3-vicuna_13B")
    parser.add_argument("--level_2_processed_json_path", required=False,
                        default="predictions/short_captions_grounded")
    parser.add_argument("--output_dir_path", required=False,
                        default="predictions/dense_captions_grounded")

    parser.add_argument("--num_processes", type=int, default=48, help="Number of concurrent processes")

    args = parser.parse_args()

    return args


def compute_similarity_spacy(sent1, sent2):
    # Get the vector representation of each phrase
    vec1 = nlp(sent1).vector
    vec2 = nlp(sent2).vector
    # Compute the cosine similarity between the two vectors
    similarity = cosine_similarity([vec1], [vec2])
    return similarity[0][0]


def get_blip2_phrases(blip2_caption):
    doc = nlp(blip2_caption.capitalize())
    nouns, phrases = [], []
    for token in doc:
        if token.pos_ == "NOUN":
            noun = token.text
            phrase = " ".join([child.text for child in token.subtree])
            if noun != phrase:
                phrase = phrase.lower().strip()
                if phrase.endswith("."):
                    phrase = phrase[:-1]
                phrases.append(phrase)
                nouns.append(noun)

    return nouns, phrases


def get_llava_phrases(llava_caption):
    doc = nlp(llava_caption.capitalize())

    all_phrase_string = ""
    nouns, phrases = [], []
    for token in doc:
        if token.pos_ == "NOUN":
            noun = token.text
            phrase_list = [child for child in token.subtree]
            if len(phrase_list) > 1:
                if len(phrase_list) == 2 and phrase_list[0].pos_ in ['DET', 'PRON'] and phrase_list[1].pos_ == 'NOUN':
                    continue
                phrase = " ".join([child.text for child in phrase_list])
                if "," in phrase:
                    sub_phrase_list = phrase.split(',')
                    sub_phrase = []
                    for sub in sub_phrase_list:
                        if noun in sub:
                            sub_phrase.append(sub.strip())
                            break
                        sub_phrase.append(sub.strip())
                    if len(sub_phrase) > 0:
                        phrase = ",".join([child for child in sub_phrase])
                if noun != phrase:
                    if phrase not in all_phrase_string:
                        all_phrase_string = all_phrase_string + " " + phrase
                        nouns.append(noun)
                        phrases.append(phrase)

    return nouns, phrases


def find_phrase_indices(phrase, caption):
    """Find start and end indices of the phrase in the caption."""
    match = re.search(re.escape(phrase.strip().lower()), caption.strip().lower())
    if match:
        return match.start(), match.end()
    return None


def is_box_inside(box1, box2):
    # Check if box1 is inside box2
    if box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]:
        return True
    return False


def correct_attribute(attribute):
    if ":" in attribute:
        attribute = ' '.join(attribute.split(':')[1:])
        attribute = attribute.strip().lower()

    return attribute


def is_plural_match(str1, str2):
    """
    Check if one string is the plural form of the other.
    """
    if str1.endswith('s') and str1[:-1] == str2:
        return True
    elif str2.endswith('s') and str2[:-1] == str1:
        return True
    return False


def get_grounding_data(phrase, caption, object_list):
    grounding_data = {"phrase": phrase, "indices": find_phrase_indices(phrase, caption), "objects": []}
    for obj in object_list:
        grounding_data["objects"].append({obj['id']: obj['bbox']})

    return grounding_data


def swap_if_needed(phrase_1, phrase_2):
    if len(phrase_1) < len(phrase_2):
        return phrase_1, phrase_2
    else:
        return phrase_2, phrase_1


def get_phrase_words(phrase):
    doc = nlp(phrase)
    # Filter out stopwords and punctuation, and get the lemma of each word
    filtered_words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    return filtered_words


def compare_phrases(phrase_1, phrase_2):
    # Phrases matches (pointing to the same concept) if one contains withing the other
    if phrase_1 in phrase_2 or phrase_2 in phrase_1:
        return 1.0
    # Count the % of same words in the phrases
    phrase_1, phrase_2 = swap_if_needed(phrase_1, phrase_2)
    phrase_1_words = get_phrase_words(phrase_1)
    hit_counts = [word in phrase_2 for word in phrase_1_words]
    if hit_counts:
        hit_percentage = sum(hit_counts) / len(hit_counts)
        # if 0.5 < hit_percentage < 0.75:
        #     similarity = compute_similarity_spacy(phrase_1, phrase_2)
        #     return similarity
        # else:
        #     return hit_percentage
        return hit_percentage
    else:
        return 0.0


def ground_caption_exact(caption, data):
    caption = caption.strip().lower()
    grounded_data = []
    all_phrases = []
    object_id_to_object_dict = {}

    # Step 1: Check for exact matches with phrases present in 'attributes'.
    for obj in data['objects'] + data['floating_objects']:
        object_id_to_object_dict[obj['id']] = obj
        if 'attributes' in obj and obj['attributes']:
            obj['attributes'] = [obj['attributes']] if isinstance(obj['attributes'], str) else obj['attributes']
            for attribute in obj['attributes']:
                attribute = correct_attribute(attribute)
                attribute = attribute.strip().lower()
                if attribute in caption:
                    all_phrases.append(attribute)
                    grounded_data.append(get_grounding_data(attribute, caption, [obj]))
                    break  # Check in the first attribute only for improving accuracy

    # Step 2: Check for exact matches with phrases present in 'relationships'.
    for grounding in data['relationships']['grounding']:
        phrase = grounding['phrase']
        phrase = phrase.strip().lower()
        if phrase in caption:
            if phrase not in all_phrases:
                obj_ids = grounding['object_ids']
                phrase_grounding_objects = [object_id_to_object_dict[obj_id] for obj_id in obj_ids]
                all_phrases.append(phrase)
                grounded_data.append(get_grounding_data(phrase, caption, phrase_grounding_objects))

    return grounded_data, all_phrases


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


def filter_groundings(caption, groundings):
    caption = traverse_and_correct(caption)
    groundings = traverse_and_correct(groundings)

    # Start prep of ann
    caption = caption.encode('utf-8').decode('unicode_escape')  # Handles unicode escape sequences
    caption = caption.translate(str.maketrans("", "", punctuation_chars))
    caption_dict = {'caption': caption.strip(), 'details': []}

    # 1. Filtering grounding to remove duplicates:
    groundings = [str(item) for item in groundings]
    # Using a dict to preserve order while removing duplicates
    unique_groundings = list({item: None for item in groundings}.keys())
    filtered_unique_groundings = [eval(item) for item in unique_groundings]

    # 2. Retaining the order of original phrases and objects, removing duplicates
    # This is done based on the priority used in generation of dense caption grounding
    seen_phrases = set()
    ordered_grounding = []
    for item in filtered_unique_groundings:
        phrase = item['phrase']
        if phrase not in seen_phrases:
            ordered_grounding.append(item)
            seen_phrases.add(phrase)

    # 3. Filtering to detect and remove the shorter overlapping phrases: Conditions:
    # A phrase is part of another phrase, The objects ids are same, The bbox are same
    ordered_grounding.sort(key=lambda x: len(x['phrase']), reverse=True)
    filtered_grounding = []
    for item in ordered_grounding:
        overlapping = False
        for existing_item in filtered_grounding:
            if item['phrase'] in existing_item['phrase'] and item['objects'] == existing_item['objects']:
                overlapping = True
                break
        if not overlapping:
            filtered_grounding.append(item)

    for relation in filtered_grounding:
        phrase = relation['phrase']
        phrase = phrase.encode('utf-8').decode('unicode_escape')  # Handles unicode escape sequences
        phrase = phrase.translate(str.maketrans("", "", punctuation_chars))
        tokens_positive = relation['indices']
        ids = []
        bboxes = []
        # considers multiple objects corresponding to a phrase
        for obj in relation['objects']:
            for id_str, bbox in obj.items():
                ids.append(int(id_str))
                bboxes.append(bbox)
        detail = {
            'phrase': phrase,
            'tokens_positive': tokens_positive,
            'ids': ids,
            'bbox': bboxes
        }
        caption_dict['details'].append(detail)

    return caption_dict


def process_file(file_names, args):
    for filename in tqdm(file_names):
        output_file_path = f"{args.output_dir_path}/{filename[:-4]}.json"
        if os.path.exists(output_file_path):
            return 1

        level_2_json_file_path = f"{args.level_2_processed_json_path}/{filename[:-4]}.json"
        level_2_json_content = json.load(open(level_2_json_file_path, 'r'))
        image_namae = list(level_2_json_content.keys())[0]
        dense_caption = open(f"{args.level_3_dense_caption_txt_dir_path}/{filename}").read().strip()

        # First, ground the caption by directly comparing attributes & phrases with it
        grounded_data, all_grounded_phrases = ground_caption_exact(dense_caption, level_2_json_content[image_namae])

        # Second, extract the phrases from the caption, and try matching each phrase with the attributes & relationships
        all_dense_caption_nouns, all_dense_caption_phrases = get_llava_phrases(dense_caption)
        all_json_attributes_and_phrases = {}  # Object ID as key - phrase/attribute as value
        object_id_to_object_dict = {}
        for obj in level_2_json_content[image_namae]['objects'] + level_2_json_content[image_namae]['floating_objects']:
            object_id_to_object_dict[obj['id']] = obj
            if obj['attributes']:
                attributes = [obj['attributes']] if isinstance(obj['attributes'], str) else obj['attributes']
                for attribute in attributes:
                    all_json_attributes_and_phrases[obj['id']] = correct_attribute(attribute)
                    break  # Check in the first attribute only for improving accuracy
        for grounding in level_2_json_content[image_namae]['relationships']['grounding']:
            phrase = grounding['phrase']
            if phrase not in all_dense_caption_phrases:
                all_json_attributes_and_phrases[grounding['object_ids'][0]] = phrase.strip().lower()
        for dense_caption_phrase in all_dense_caption_phrases:
            dense_caption_phrase = dense_caption_phrase.strip().lower()
            if dense_caption_phrase not in all_grounded_phrases:
                all_candidates = []
                all_candidates_score = []
                all_candidates_object_id = []
                for object_id in all_json_attributes_and_phrases.keys():
                    json_phrase = all_json_attributes_and_phrases[object_id]
                    score = compare_phrases(dense_caption_phrase, json_phrase)
                    if score >= 0.75:
                        all_candidates.append(json_phrase)
                        all_candidates_score.append(score)
                        all_candidates_object_id.append(object_id)
                if all_candidates_score:
                    selected_object_id = all_candidates_object_id[np.argmax(np.array(all_candidates_score))]
                    grounded_data.append(get_grounding_data(dense_caption_phrase, dense_caption,
                                                            [object_id_to_object_dict[selected_object_id]]))
                    all_grounded_phrases.append(dense_caption_phrase)

        # Third, collect the un-grounded phrases, extract nouns and try matching it with objects
        ungrounded_nouns_and_phrases = [(noun, phrase) for noun, phrase in
                                        zip(all_dense_caption_nouns, all_dense_caption_phrases) if
                                        phrase not in all_grounded_phrases]
        # Create a dictionary with keys as object ids and values as labels
        object_id_to_labels = {}
        for obj in level_2_json_content[image_namae]['objects']:
            object_id_to_labels[obj['id']] = obj['labels']
        for _, dense_caption_phrase in ungrounded_nouns_and_phrases:
            nouns, phrases = get_blip2_phrases(dense_caption_phrase)
            for noun, phrase in zip(nouns, phrases):
                found = []
                for obj_id, labels in object_id_to_labels.items():
                    if noun in labels:
                        found.append(obj_id)
                    else:
                        for label in labels:
                            if is_plural_match(noun, label):
                                found.append(obj_id)
                if len(found) == 1:  # TODO: Ground phrase within dense_caption_phrase to improve accuracy
                    grounded_data.append(get_grounding_data(phrase, dense_caption,
                                                            [object_id_to_object_dict[found[0]]]))
                    all_grounded_phrases.append(phrase)
                elif len(found) > 1:
                    if len(found) == 2:
                        # Check if one box is inside the other, if so select the bigger box
                        shortlisted_objects = [object_id_to_object_dict[id] for id in found]
                        selected_object = None
                        if is_box_inside(shortlisted_objects[0]['bbox'], shortlisted_objects[1]['bbox']):
                            selected_object = shortlisted_objects[1]
                        elif is_box_inside(shortlisted_objects[1]['bbox'], shortlisted_objects[0]['bbox']):
                            selected_object = shortlisted_objects[0]
                        if selected_object:
                            grounded_data.append(get_grounding_data(phrase, dense_caption, [selected_object]))
                            all_grounded_phrases.append(phrase)
                    else:
                        pass  # May be try grounding using background, foreground, etc. information along with depth values
                else:
                    # The noun does not exists in objects
                    pass
        # Save the updated json with dense caption and grounding
        dense_caption_groundings = filter_groundings(dense_caption, grounded_data)
        level_2_json_content[image_namae]['dense_caption'] = dense_caption_groundings

        with open(output_file_path, 'w') as f:
            json.dump(level_2_json_content, f, indent=4)


def split_list(input_list, n):
    """Split a list into 'n' parts using numpy."""
    arrays = np.array_split(np.array(input_list), n)
    return [arr.tolist() for arr in arrays]


def main():
    args = parse_args()
    os.makedirs(args.output_dir_path, exist_ok=True)
    all_dense_cap_txt_files = os.listdir(args.level_3_dense_caption_txt_dir_path)

    # Combine filename and args into a tuple
    all_dense_cap_txt_files_list = split_list(all_dense_cap_txt_files, n=args.num_processes)
    task_args = [(raw_file, args) for raw_file in all_dense_cap_txt_files_list]
    with Pool() as pool:
        pool.starmap(process_file, task_args)


if __name__ == "__main__":
    main()
