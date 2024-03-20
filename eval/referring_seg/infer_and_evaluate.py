import os
import sys
import json
import tqdm
import torch
import argparse
import deepspeed
import transformers
from functools import partial
from torch.utils.data import ConcatDataset
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from dataset.dataset import custom_collate_fn
from dataset.segm_datasets.RefCOCO_Segm_ds import ReferSegmDataset
from tools.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, AverageMeter, Summary,
                         intersectionAndUnionGPU, dict_to_cuda)


def parse_args(args):
    parser = argparse.ArgumentParser(description="GLaMM Model Evaluation")

    # Model-specific settings
    parser.add_argument("--version", required=True, help="Path to the pretrained model for evaluation.")
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true", default=False)
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true", default=False)
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--with_region", action="store_true", default=True)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default="", type=str)
    parser.add_argument("--precision", default='bf16', type=str)

    # Training settings
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=2, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)

    # Dataset settings
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--refer_seg_data", default="refcocog|val", type=str)
    parser.add_argument("--results_path", default="referring_seg_eval.json", type=str)

    # Evaluation settings
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")

    # Experiment settings
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="glam_eval_referseg", type=str)

    return parser.parse_args(args)


def initialize_environment(args):
    """ Set up logging and model directories. """
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        return SummaryWriter(args.log_dir)
    return None


def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token

    if not args.pretrained:
        if args.use_mm_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        # modifications specific for regions
        reg_tokens = ['<bbox>', '<point>']
        # Adding special tokens for pixel grounding
        segmentation_tokens = ['[SEG]']
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        special_tokens = reg_tokens + segmentation_tokens + phrase_tokens
        tokenizer.add_tokens(special_tokens, special_tokens=True)

    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[0]
    args.eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[0]

    return tokenizer


def initialize_model(args, tokenizer):
    """ Initialize the GLaMM model. """
    model_args = {k: getattr(args, k) for k in
                  ["train_mask_decoder", "out_dim", "ce_loss_weight", "dice_loss_weight", "bce_loss_weight",
                   "seg_token_idx", "vision_pretrained", "vision_tower", "use_mm_start_end", "mm_vision_select_layer",
                   "pretrain_mm_mlp_adapter", "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", "mm_use_im_start_end",
                   "with_region", "bbox_token_idx", "eop_token_idx", "bop_token_idx"]}
    model_args["num_level_reg_features"] = 4

    model = GLaMMForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args
    )
    print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def prepare_model_for_training(model, tokenizer, args):
    # Enable input gradients
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Initialize vision tower
    print(
        '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
            args.vision_tower
        ) + '\033[0m'
    )
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=args.local_rank)

    # Initialize GLaMM model and adjust requires_grad
    if not args.pretrained:
        model.get_model().initialize_glamm_model(model.get_model().config)
    else:
        for param in model.get_model().grounding_encoder.parameters():
            param.requires_grad = False
        if model.get_model().config.train_mask_decoder:
            model.get_model().grounding_encoder.mask_decoder.train()
            for param in model.get_model().grounding_encoder.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        model.get_model().text_hidden_fcs.train()
        for param in model.get_model().text_hidden_fcs.parameters():
            param.requires_grad = True

    # Set requires_grad for vision tower and mm projector
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # Set requires_grad based on LoRA training
    lora_r = args.lora_r
    if lora_r == 0:
        for p in model.get_model().layers.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # Configure LoRA if applicable
    if lora_r > 0:
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))


def setup_lora_config(model, args):
    """ Configure LoRA settings for the model. """

    def find_proj_layers(model, target_modules):
        """ Identify projection layers in the model for LoRA adaptation. """
        linear_cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if (isinstance(module, linear_cls) and all(
                    x not in name for x in ["grounding_encoder", "vision_tower", "mm_projector", "text_hidden_fcs"]
            ) and any(x in name for x in target_modules)):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    # Extracting LoRA target modules
    lora_target_modules = args.lora_target_modules.split(",")
    lora_module_names = find_proj_layers(model, lora_target_modules)

    # Configuring LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=lora_module_names, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM"
    )
    return lora_config


def initialize_datasets_and_loaders(args, tokenizer):
    # Dataset settings for ReferSegDataset
    common_ds_args = {
        "dataset_dir": args.dataset_dir,
        "tokenizer": tokenizer,
        "global_image_encoder": args.vision_tower,
        "precision": args.precision,
        "image_size": args.image_size
    }

    # Validation datasets
    dataset, split = args.refer_seg_data.split('|')
    val_datasets = [ReferSegmDataset(**common_ds_args, validation=True, refer_segm_data=dataset, split=split,
                                     inference=True)]
    _ = [d._set_len(len(d.refer_segm_data[dataset]['images'])) for d in val_datasets]

    return val_datasets


def setup_data_loaders(args, val_datasets, tokenizer):
    sampler_args = {"shuffle": False, "drop_last": False}
    val_loader_args = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": args.workers,
                       "pin_memory": False}
    collate_fn_args_val = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=True
    )

    # Validation loader
    combined_val_datasets = ConcatDataset(val_datasets)
    val_loader = torch.utils.data.DataLoader(
        combined_val_datasets, **val_loader_args, collate_fn=collate_fn_args_val,
        sampler=torch.utils.data.distributed.DistributedSampler(combined_val_datasets, **sampler_args), )

    return val_loader


def initialize_deepspeed(model, tokenizer, args):
    ds_config = {"train_micro_batch_size_per_gpu": args.batch_size,
                 "gradient_accumulation_steps": args.grad_accumulation_steps, "optimizer": {"type": "AdamW",
                                                                                            "params": {"lr": args.lr,
                                                                                                       "weight_decay": 0.0,
                                                                                                       "betas": (
                                                                                                           args.beta1,
                                                                                                           args.beta2)}},
                 "scheduler": {"type": "WarmupDecayLR",
                               "params": {"total_num_steps": args.epochs * args.steps_per_epoch, "warmup_min_lr": 0,
                                          "warmup_max_lr": args.lr, "warmup_num_steps": 100, "warmup_type": "linear"}},
                 "fp16": {"enabled": args.precision == "fp16"}, "bf16": {"enabled": args.precision == "bf16"},
                 "gradient_clipping": 1.0,
                 "zero_optimization": {"stage": 2, "contiguous_gradients": True, "overlap_comm": True,
                                       "reduce_scatter": True, "reduce_bucket_size": 5e8,
                                       "allgather_bucket_size": 5e8}, }

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), collate_fn=partial(
            custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank
        ), config=ds_config
    )

    return model_engine, optimizer, scheduler


def evaluate_model_performance(validation_loader, model, args):
    # Trackers for metrics
    trackers = {
        "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
        "union": AverageMeter("Union", ":6.3f", Summary.SUM),
        "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
    }

    model.eval()
    for data_batch in tqdm.tqdm(validation_loader):
        # Prepare data and convert relevant tensors to the appropriate type
        data_batch = dict_to_cuda(data_batch)
        for key in ["global_enc_images", "grounding_enc_images"]:
            data_batch[key] = data_batch[key].to(dtype=torch.bfloat16, device=args.local_rank)

        torch.cuda.empty_cache()

        # Model inference without gradient tracking
        with torch.no_grad():
            results = model(**data_batch)

        predictions = results["pred_masks"]
        gt_masks = results["gt_masks"][0].int()
        predicted_masks = (predictions[0] > 0).int()  # Thresholding to get binary masks
        assert len(predictions) == 1

        intersection, union, accuracy_iou = 0.0, 0.0, 0.0
        for target, prediction in zip(gt_masks, predicted_masks):
            intersect, union_, _ = intersectionAndUnionGPU(
                prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
            )
            intersection += intersect
            union += union_
            accuracy_iou += intersect / (union_ + 1e-5)
            # handles no-object targets
            accuracy_iou[union_ == 0] += 1.0

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        accuracy_iou = accuracy_iou.cpu().numpy() / gt_masks.shape[0]
        trackers["intersection"].update(intersection)
        trackers["union"].update(union)
        trackers["gIoU"].update(accuracy_iou, n=gt_masks.shape[0])

    for meter in trackers.values():
        meter.all_reduce()

    iou_per_class = trackers["intersection"].sum / (trackers["union"].sum + 1e-10)
    class_iou = iou_per_class[1]
    global_iou = trackers["gIoU"].avg[1]

    return global_iou, class_iou


def main(args):
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    prepare_model_for_training(model, tokenizer, args)

    model_engine, _, _ = initialize_deepspeed(model, tokenizer, args)

    val_datasets = initialize_datasets_and_loaders(args, tokenizer)
    val_loader = setup_data_loaders(args, val_datasets, tokenizer)

    giou, ciou = evaluate_model_performance(val_loader, model_engine, args)

    torch.distributed.barrier()
    if args.local_rank == 0:
        # Update and save the results
        os.makedirs(args.results_path, exist_ok=True)
        if os.path.exists(f"{args.results_path}/stats.json"):
            with open(f"{args.results_path}/stats.json", 'r') as json_file:
                result_list = json.load(json_file)
        else:
            result_list = []
        result_dict = {"model": args.results_path, "dataset": args.refer_seg_data, "giou": str(giou), "ciou": str(ciou)}
        result_list.append(result_dict)

        with open(f"{args.results_path}/stats.json", 'w') as json_file:
            json.dump(result_list, json_file, indent=2)

        print(result_list)  # Print all the results


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
