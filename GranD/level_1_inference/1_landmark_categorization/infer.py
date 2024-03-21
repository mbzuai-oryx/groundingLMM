import json
import os
import argparse
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import numpy as np
import requests
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO
from tqdm import tqdm
import multiprocessing
from random import shuffle
from multiprocessing.pool import Pool
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--image_dir_path", required=True,
                        help="Path to the directory containing images.")
    parser.add_argument("--output_dir_path", required=True,
                        help="Path to the output directory to store the predictions.")
    parser.add_argument("--gpu_ids", required=True,
                        help="Comma separated list of gpu_ids to run the inference on. "
                             "For example '0,1,2,3' will run the inference on the 4 GPUs.")
    parser.add_argument("--llava_model_path", required=True,
                        help="Path to the llava model - llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3")

    args = parser.parse_args()

    return args


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def get_prompt(model, qs, conv_mode="llava_v1", ):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    return prompt, stop_str


def get_main_prompt(model, conv_mode="llava_v1"):
    options = ["Indoor scene", "Outdoor scene", "Transportation scene", "Sports and recreation scene"]
    qs = (f"Categorize the image landmark into one of the following options:\n"
          f"1. {options[0]}\n"
          f"2. {options[1]}\n"
          f"3. {options[2]}\n"
          f"4. {options[3]}\n"
          f"Respond with only the option.")

    return get_prompt(model, qs, conv_mode)


def get_fine_prompt(model, landmark_category, conv_mode="llava_v1"):
    if landmark_category == "Indoor scene":
        options = ["Living space", "Work space", "Public space", "Industrial space"]
        qs = (f"Categorize the image landmark into one of the following {landmark_category}s:\n"
              f"1. {options[0]}\n"
              f"2. {options[1]}\n"
              f"3. {options[2]}\n"
              f"4. {options[3]}\n"
              f"Respond with only the option.")
    elif landmark_category == "Outdoor scene":
        options = ["Urban landscape", "Rural landscape", "Natural landscape"]
        qs = (f"Categorize the image landmark into one of the following {landmark_category}s:\n"
              f"1. {options[0]}\n"
              f"2. {options[1]}\n"
              f"3. {options[2]}\n"
              f"Respond with only the option.")
    elif landmark_category == "Transportation scene":
        options = ["Road", "Airport", "Train station", "Port and harbor"]
        qs = (f"Categorize the image landmark into one of the following {landmark_category}s:\n"
              f"1. {options[0]}\n"
              f"2. {options[1]}\n"
              f"3. {options[2]}\n"
              f"4. {options[3]}\n"
              f"Respond with only the option.")
    elif landmark_category == "Sports and recreation scene":
        options = ["Sporting venue", "Recreational area", "Gym and fitness center"]
        qs = (f"Categorize the image landmark into one of the following {landmark_category}s:\n"
              f"1. {options[0]}\n"
              f"2. {options[1]}\n"
              f"3. {options[2]}\n"
              f"Respond with only the option.")
    else:
        qs = ""

    return get_prompt(model, qs, conv_mode)


def get_fine_category(main_category, outputs):
    if main_category == "Indoor scene":
        fine_category = "Living space" if "living space" in outputs \
            else "Work space" if "work space" in outputs \
            else "Public space" if "public space" in outputs \
            else "Industrial space" if "industrial space" in outputs \
            else ""
    elif main_category == "Outdoor scene":
        fine_category = "Urban landscape" if "urban landscape" in outputs \
            else "Rural landscape" if "rural landscape" in outputs \
            else "Natural landscape" if "natural landscape" in outputs \
            else ""
    elif main_category == "Transportation scene":
        fine_category = "Road" if "road" in outputs \
            else "Airport" if "airport" in outputs \
            else "Train station" if "train station" in outputs \
            else "Port and harbor" if "port and harbor" in outputs \
            else ""
    elif main_category == "Sports and recreation scene":
        fine_category = "Sporting venue" if "sporting venue" in outputs \
            else "Recreational area" if "recreational area" in outputs \
            else "Gym and fitness center" if "gym and fitness center" in outputs \
            else ""
    else:
        fine_category = ""

    return fine_category


def run_inference(llava_model_path, image_dir_path, image_names, output_dir_path, gpu_id):
    torch.cuda.set_device(gpu_id)
    # Create lava model
    model_name = get_model_name_from_path(llava_model_path)
    model = load_pretrained_model(llava_model_path, None, model_name, device_map=gpu_id)

    tokenizer, model, image_processor, context_len = model
    start_time = time.time()
    for image_name in tqdm(image_names):
        json_file_path = f"{output_dir_path}/landmark/{image_name[:-4]}.json"
        # Check if the file already exists
        if os.path.exists(json_file_path):
            continue  # Skip this image since it's already processed

        image_path = f"{image_dir_path}/{image_name}"
        # Load image
        image = load_image(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        # Prepare the prompt for main landmark
        main_prompt, stop_str = get_main_prompt(model)
        input_ids = tokenizer_image_token(main_prompt, tokenizer,
                                          IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # Run LLaVA inference
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        # Post process
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs_main = outputs.strip().lower()
        landmark_category = "Indoor scene" if "indoor scene" in outputs_main \
            else "Outdoor scene" if "outdoor scene" in outputs_main \
            else "Transportation scene" if "transportation scene" in outputs_main \
            else "Sports and recreation scene" if "sports and recreation scene" in outputs_main \
            else ""



        # Prepare the prompt for find landmark
        fine_prompt, stop_str = get_fine_prompt(model, landmark_category)
        input_ids = tokenizer_image_token(fine_prompt, tokenizer,
                                          IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # Run LLaVA inference
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        # Post process
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs_fine = outputs.strip().lower()
        landmark_fine_category = get_fine_category(main_category=landmark_category, outputs=outputs_fine)

        all_data = {}
        all_data[image_name] = {}
        all_data[image_name]['landmark'] = {"category": landmark_category, "fine_category": landmark_fine_category}
        with open(json_file_path, 'w') as f:
            json.dump(all_data, f)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- Landmark Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')


def split_list(input_list, n):
    """Split a list into 'n' parts using numpy."""
    shuffle(input_list)
    arrays = np.array_split(np.array(input_list), n)
    return [arr.tolist() for arr in arrays]


def main():
    args = parse_args()
    image_dir_path = args.image_dir_path
    output_dir_path = args.output_dir_path
    llava_model_path = args.llava_model_path
    gpu_ids = args.gpu_ids

    os.makedirs(output_dir_path, exist_ok=True)
    os.makedirs(f"{output_dir_path}/landmark", exist_ok=True)

    gpu_ids = gpu_ids.split(',')
    gpu_ids = [int(id) for id in gpu_ids]
    num_processes = len(gpu_ids)

    # Create cfgs for all the models
    image_names_list = split_list(os.listdir(image_dir_path), num_processes)
    task_args = [(llava_model_path, image_dir_path, image_names, output_dir_path, id)
                 for image_names, id in zip(image_names_list, gpu_ids)]

    # Use a pool of workers to process the files in parallel.
    with Pool() as pool:
        pool.starmap(run_inference, task_args)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
