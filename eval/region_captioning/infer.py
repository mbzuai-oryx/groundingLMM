import re
import cv2
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor
from torch.utils.data import DataLoader, DistributedSampler

from eval.utils import *
from eval.ddp import *
from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Inference - Region Captioning")

    parser.add_argument("--hf_model_path", required=True, help="The model path in huggingface format.")
    parser.add_argument("--annotation_file",
                        default="data/RefCoco_Reg/mdetr_annotations/finetune_refcocog_val_captions.json", type=str,
                        help="Replace with 'data/visual_genome/test_caption.json' for VG.")
    parser.add_argument("--image_dir", default="data/coco_2014/train2014", type=str,
                        help="Replace with 'data/visual_genome/images' for VG")
    parser.add_argument("--dataset", default="refcocog", type=str, help="Options are 'refcocog', 'vg'")
    parser.add_argument("--results_dir", default="results", type=str, help="The path to save the results.")


    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"], )

    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def inference(instructions, inputs):
    # Extract the inputs
    bbox_img = inputs['boxes']
    image_path = inputs['image']

    instructions = instructions.replace('&lt;', '<').replace('&gt;', '>')

    # Prepare prompt for model Inference
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
    prompt = begin_str + instructions
    if args.use_mm_start_end:
        replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    # Read and preprocess the image (Global image encoder - CLIP)
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]
    image_clip = (clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda())
    image_clip = image_clip.bfloat16()  # Precision is bf16 by default

    # Preprocess the image (Grounding image encoder)
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = (
        grounding_image_ecoder_preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda())
    image = image.bfloat16()  # Precision is bf16 by default

    # Prepare inputs for inference
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()
    bboxes = None
    if len(bbox_img) > 0:
        height, width = original_size_list[0]  # Original Image Dimensions

        # Rescaling BBox to 336*336
        x_scale, y_scale = 336 / width, 336 / height
        bboxes_scaled = [[bbox[0] * x_scale, bbox[1] * y_scale,
                          bbox[2] * x_scale, bbox[3] * y_scale] for bbox in bbox_img]
        ori_bboxes = np.array(bboxes_scaled, dtype=np.float64)
        height_sc, width_sc = (336, 336)  # To normalize the Image
        norm_bboxes = ori_bboxes / np.array([width_sc, height_sc, width_sc, height_sc])
        bboxes = [torch.tensor(norm_bboxes).cuda().half().to(torch.bfloat16)]

    # Generate output
    output_ids, pred_masks = model.evaluate(image_clip, image, input_ids, resize_list, original_size_list,
                                            max_tokens_new=512, bboxes=bboxes)
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    # Post-processing
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    cleaned_str = re.sub(r'<.*?>', '', text_output)

    # Remove the [SEG] token
    cleaned_str = cleaned_str.replace('[SEG]', '')

    # Strip unnecessary spaces
    cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()

    return cleaned_str


def custom_collate_fn(batch):
    image_id = [item[0] for item in batch]
    filename = [item[1] for item in batch]
    bbox = [item[2] for item in batch]
    gt = [item[3] for item in batch]

    return image_id, filename, bbox, gt


if __name__ == "__main__":
    args = parse_args()
    init_distributed_mode(args)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, cache_dir=None,
                                              model_max_length=args.model_max_length, padding_side="right",
                                              use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    torch_dtype = torch.bfloat16  # By default, using bf16
    kwargs = {"torch_dtype": torch_dtype}
    model = GLaMMForCausalLM.from_pretrained(args.hf_model_path, low_cpu_mem_usage=True,
                                             seg_token_idx=seg_token_idx, **kwargs)
    # Update model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Initialize Global Image Encoder (CLIP)
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    # Transfer the model to GPU
    model = model.bfloat16().cuda()  # Replace with model = model.float().cuda() for 32 bit inference
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device="cuda")

    # Initialize Image Processor for GLobal Image Encoder (CLIP)
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()  # Model should be in evaluation mode for inference

    # Prompt model to perfor region captioning task
    instruction = "Can you provide me with a detailed description of the region in the picture marked by <bbox>?"

    # Intermediate results path is hard-coded (you may change it as per your needs)
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = f"{args.results_dir}/{os.path.basename(args.hf_model_path)}_{args.dataset}_{args.rank}.json"

    # Create DDP Dataset
    dataset = RegionCapDDP(args.annotation_file)
    distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, num_workers=2,
                            sampler=distributed_sampler, collate_fn=custom_collate_fn)

    # Iterate over all the samples, perform inference and save results
    results = []
    for idx, (image_id, filename, bbox, gt) in enumerate(tqdm(dataloader)):
        image_id, filename, bbox, gt = image_id[0], filename[0], bbox[0], gt[0]
        image_path = os.path.join(args.image_dir, filename)
        inputs = {'image': image_path, 'boxes': [bbox]}

        result_caption = inference(instruction, inputs)  # Perform inference

        result_dict = {}
        result_dict["image_id"] = image_id
        result_dict["caption"] = result_caption
        results.append(result_dict)

    with open(results_path, 'w') as json_file:
        json.dump(results, json_file, indent=2)
