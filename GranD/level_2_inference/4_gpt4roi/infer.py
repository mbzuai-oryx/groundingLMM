import argparse
import os.path
from functools import partial
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel
from ddp import *
from inference_utils import *
from torch.utils.data import DataLoader, DistributedSampler
import torch
from gpt4roi.models.spi_llava import SPILlavaMPTForCausalLM
from llava.model.utils import KeywordsStoppingCriteria
from llava.utils import disable_torch_init


def custom_collate_fn(batch):
    image_names = [item[0] for item in batch]
    image_size = [item[1] for item in batch]
    init_inputs = [item[2] for item in batch]
    filtered_object_ids = [item[3] for item in batch]
    json_content = [item[4] for item in batch]

    return image_names, image_size, init_inputs, filtered_object_ids, json_content


def main(args):
    init_distributed_mode(args)
    os.makedirs(args.output_dir_path, exist_ok=True)

    # Create model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SPILlavaMPTForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True,
                                                   torch_dtype=torch.float16, use_cache=True).cuda()
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)
    mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                         special_tokens=True)
    spi_tokens = ['<bbox>', '<point>']
    tokenizer.add_tokens(spi_tokens, special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]
    if vision_tower.device.type == 'meta':
        vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower.config._name_or_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True).cuda()
        model.get_model().vision_tower[0] = vision_tower
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)
    vision_tower.to(device='cuda', dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = \
        tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    stop_str = '###'
    keywords = [stop_str]
    model.model.tokenizer = tokenizer

    # Create dataset
    image_dataset = CustomJsonDataset(args.image_dir_path,
                                      args.level_2_pred_path, image_processor, tokenizer)
    distributed_sampler = DistributedSampler(image_dataset, rank=args.rank, shuffle=False)
    dataloader = DataLoader(image_dataset, batch_size=args.batch_size_per_gpu, num_workers=4,
                            sampler=distributed_sampler, collate_fn=custom_collate_fn)
    # Run on all level-1-processed files
    for (image_name, image_size, init_inputs, filtered_object_ids, json_content) in tqdm(dataloader):
        image_name, image_size, init_inputs, filtered_object_ids, json_content = (image_name[0], image_size[0],
                                                                                  init_inputs[0],
                                                                                  filtered_object_ids[0],
                                                                                  json_content[0])
        output_json_path = os.path.join(args.output_dir_path, f"{image_name[:-4]}.json")
        if os.path.exists(output_json_path):
            continue
        image = init_inputs['image']
        input_ids = init_inputs['input_ids'].cuda()[None]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        for object in json_content[image_name]['objects']:
            if object['id'] in filtered_object_ids:
                pred = object['bbox']
                attributes = []
                if object['attributes']:
                    attributes.append(object['attributes'][0])

                torch_pred_bboxes = get_bboxes(image_size, pred)
                bboxes = torch_pred_bboxes.cuda()
                with torch.inference_mode():

                    model.orig_forward = model.forward
                    model.forward = partial(model.orig_forward,
                                            img_metas=[None],
                                            bboxes=[bboxes.half()])

                    with torch.amp.autocast(device_type='cuda'):
                        output_ids = model.generate(
                            input_ids,
                            images=image.unsqueeze(0).half().cuda(),
                            do_sample=True,
                            temperature=0.2,
                            max_new_tokens=1024,
                            stopping_criteria=[stopping_criteria])
                    model.forward = model.orig_forward

                input_token_len = input_ids.shape[1]
                n_diff_input_output = (
                        input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(
                        f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:],
                                                 skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                output = outputs.strip()
                attributes.insert(0, output.split(': ')[1])
                # attributes.append(output.split(': ')[1])
                object['attributes'] = attributes

        add_big_bboxes(image_size, json_content[image_name]['objects'], json_content[image_name]['floating_objects'])
        with open(output_json_path, 'w') as f:
            json.dump(json_content, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', required=False, default='GPT4RoI-7B')
    parser.add_argument('--image_dir_path', required=True)
    parser.add_argument('--level_2_pred_path', required=False, default='predictions/level-2-processed')
    parser.add_argument('--output_dir_path', required=False,
                        default='predictions/level-2-processed_gpt4roi')

    # DDP related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    main(args)
