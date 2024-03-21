import argparse
import os
import json
from tqdm import tqdm
from prompt_template import get_prompt, get_simple_level_4_scene_graph
from vllm import LLM, SamplingParams
from fastchat.model import get_conversation_template
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--image_names_txt_path", required=True)
    parser.add_argument("--model_path", required=False, default="vicuna-13b-v1.5",
                        help="Path to vicuna-13b-v1.5.")
    parser.add_argument("--level_2_dir_path", required=False,
                        default="predictions/level-2-processed_labelled")
    parser.add_argument("--output_directory_path", required=False,
                        default="predictions/level-4-vicuna-13B")

    parser.add_argument("--batch_size", required=False, type=int, default=32)

    args = parser.parse_args()

    return args


def get_chat_completion_prompt(model_path, scene_graph):
    prompt = get_prompt(scene_graph)
    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt


def run_vicuna_inferene(model_path, model, sampling_params, scene_graphs):
    prompts = [get_chat_completion_prompt(model_path, scene_graph) for scene_graph in scene_graphs]

    outputs = model.generate(prompts, sampling_params, use_tqdm=False)
    responses, reasons = [], []
    for output in outputs:
        responses.append(output.outputs[0].text)
        reasons.append(output.outputs[0].finish_reason)

    return responses, reasons


def main():
    args = parse_args()
    # Create output directory path if not exists
    output_path = f"{args.output_directory_path}"
    os.makedirs(output_path, exist_ok=True)

    # Create sampling params & load Vicuna Model
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
    model = LLM(model=args.model_path, max_num_batched_tokens=4096)

    with open(args.image_names_txt_path, 'r') as f:
        image_names = [line.strip() for line in f if line.strip()]
    total_batch_size = min(args.batch_size, len(image_names))
    start_time = time.time()
    processed_images = set(os.listdir(output_path))
    for i in tqdm(range(0, len(image_names), total_batch_size)):
        batch_image_names = image_names[i:i + total_batch_size]
        filtered_batch_image_names = []
        batch_scene_graphs = []
        batch_level_2_json_contents = {}

        # Prepare batch data and check if already processed
        for image_name in batch_image_names:
            if f"{image_name[:-4]}.txt" in processed_images:
                continue

            level_2_json_contents = json.load(open(f"{args.level_2_dir_path}/{image_name[:-4]}.json", 'r'))

            try:
                scene_graph = get_simple_level_4_scene_graph(image_name, level_2_json_contents)
            except Exception as e:
                print(f"Error while finding the scene graph for {image_name}")
                continue

            filtered_batch_image_names.append(image_name)
            batch_scene_graphs.append(scene_graph)
            batch_level_2_json_contents[image_name] = level_2_json_contents

        if len(batch_scene_graphs) == 0:
            continue

        dense_captions, vllm_stop_reasons = run_vicuna_inferene(args.model_path, model, sampling_params,
                                                                batch_scene_graphs)

        if not len(vllm_stop_reasons) == len(filtered_batch_image_names):
            print(f"Error processing batch starting from {filtered_batch_image_names[0]}. "
                  f"Skipping whole batch to ensure the correctness.")
            continue

        for idx, image_name in enumerate(filtered_batch_image_names):
            if not vllm_stop_reasons[idx] == 'stop':  # TODO: List index out of range error needs to be fixed.
                print(f"Processing of image: {image_name} failed. Reasons: {vllm_stop_reasons[idx]}")
                continue

            output_json_path = f"{output_path}/{image_name[:-4]}.txt"
            with open(output_json_path, 'w') as f:
                f.write(dense_captions[idx])

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- level-3 Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')


if __name__ == "__main__":
    main()
