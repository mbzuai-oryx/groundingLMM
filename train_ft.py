"""
train_ft.py - GLaMM Training on Single Dataset Type

Trains the GLaMM model on one dataset type (Caption, Region, or Segmentation) at a time, iterating thoroughly through
the chosen dataset. This targeted approach is optimal for specialized training on specific downstream task.
"""
import os
import sys
import time
import tqdm
import random
import torch
import argparse
import deepspeed
import numpy as np
import transformers
from functools import partial
from torch.utils.data import ConcatDataset
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib

from dataset.dataset import custom_collate_fn
from tools.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, AverageMeter, ProgressMeter, dict_to_cuda,
                         Summary, intersectionAndUnionGPU)

from dataset.gcg_datasets.GranDf_gcg_ds import GranDfDataset, OpenPsgGCGDataset, Flickr30kGCGDataset, RefCOCOgGCGDataset
from dataset.caption_datasets.COCO_Caption_ds import CocoCapDataset
from dataset.caption_datasets.LLavaInstruct_vqa_ds import LLaVAInstructDataset
from dataset.region_datasets.RefCOCO_VG_Region_ds import (RefCocoRegDataset, RefCocoGRegDataset, RefCocoPRegDataset,
                                                       VisualGenomeRegDataset)
from dataset.region_datasets.Flickr_Region_ds import Flickr30kRegDataset
from dataset.segm_datasets.Semantic_Segm_ds import SemanticSegmDataset
from dataset.segm_datasets.RefCOCO_Segm_ds import ReferSegmDataset


def parse_args(args):
    parser = argparse.ArgumentParser(description="GLaMM Model Training")

    # Model-specific settings
    parser.add_argument("--version", default="MBZUAI/GLaMM-GranD-Pretrained")
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true")
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true")
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--with_region", action="store_true", default=True)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default="", type=str)
    parser.add_argument("--precision", default='bf16', type=str)

    # Dataset settings
    parser.add_argument("--use_cap_data", action="store_true", help="Use caption data")
    parser.add_argument("--use_reg_data", action="store_true", help="Use region data")
    parser.add_argument("--use_segm_data", action="store_true", help="Use segmentation data")
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--seg_dataset", default="Semantic_Segm||Refer_Segm||RefCoco_GCG||PSG_GCG||Flickr_GCG||GranDf_GCG",
                        type=str, help="Choose from: Semantic_Segm, Refer_Segm, RefCoco_GCG, GranDf_GCG, PSG_GCG, Flickr_GCG")
    parser.add_argument("--segm_sample_rates", default="5,4,3,3,3,1", type=str)
    parser.add_argument("--reg_dataset", default="RefCoco_Reg||RefCocoG_Reg||RefCocoP_Reg||VisGen_Reg",
                        type=str, help="Choose from: RefCoco_Reg, RefCocoG_Reg, RefCocoP_Reg, VisGen_Reg, Flickr_Reg")
    parser.add_argument("--reg_sample_rates", default="1,1,1,1", type=str)
    parser.add_argument("--cap_dataset", default="CocoCap||LLaVaInstruct", type=str, help="Choose from: CocoCap, LLaVaInstruct")
    parser.add_argument("--cap_sample_rates", default="1,1", type=str)
    parser.add_argument("--semantic_segm_data", default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary", type=str)
    parser.add_argument("--refer_segm_data", default="refcoco||refcoco+||refcocog||refclef", type=str)
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)

    # Training settings
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=2, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=2, type=int)
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
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")

    # Evaluation settings
    parser.add_argument("--val_dataset", default="RefCOCOgRegVal", type=str,
                        help="Choose from: CocoCapVal, RefCOCOgRegVal, VisGenomeRegVal, RefCOCOgSegmVal, PsgGCGVal, "
                             "RefCocoGCGVal, FlickrGCGVal")
    parser.add_argument("--mask_validation", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    # Experiment settings
    parser.add_argument("--log_base_dir", default="./output", type=str)
    parser.add_argument("--exp_name", default="GlamFinetuneOS", type=str)

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

    # Make certain modules trainable
    set_trainable_modules(model)


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


def set_trainable_modules(model):
    """ Make specified modules in the model trainable. """
    trainable_modules = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "region_encoder"]
    for name, param in model.named_parameters():
        if any(module in name for module in trainable_modules):
            print(f"Making trainable: {name}, Shape: {param.shape}")
            param.requires_grad = True

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('\033[92m' + "---- Total parameters: ----{}".format(total_params) + '\033[0m')
        print('\033[92m' + "---- Trainable parameters: ----{}".format(trainable_params) + '\033[0m')

    count_parameters(model)


def initialize_datasets_and_loaders(args, tokenizer):
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    # Common dataset arguments
    common_ds_args = {"dataset_dir": args.dataset_dir, "tokenizer": tokenizer,
                      "global_image_encoder": args.vision_tower,
                      "epoch_samples": args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size,
                      "precision": args.precision, "image_size": args.image_size,
                      "num_classes_per_sample": args.num_classes_per_sample}

    cap_dataset_classes = {"CocoCap": CocoCapDataset,
                           "LLaVaInstruct": LLaVAInstructDataset,
                           }
    reg_dataset_classes = {"RefCoco_Reg": RefCocoRegDataset,
                           "RefCocoG_Reg": RefCocoGRegDataset,
                           "RefCocoP_Reg": RefCocoPRegDataset,
                           "VisGen_Reg": VisualGenomeRegDataset,
                           "Flickr_Reg": Flickr30kRegDataset,
                           }
    seg_dataset_classes = {"Semantic_Segm": SemanticSegmDataset,
                           "Refer_Segm": ReferSegmDataset,
                           "PSG_GCG": OpenPsgGCGDataset,
                           "RefCoco_GCG": RefCOCOgGCGDataset,
                           "GranDf_GCG": GranDfDataset,
                           "Flickr_GCG": Flickr30kGCGDataset,
                          }
    # Train datasets
    if args.use_cap_data:
        train_datasets = [cap_dataset_classes[ds_name](**common_ds_args, random_sampling=False)
                          for ds_name in args.cap_dataset.split("||")]
    elif args.use_reg_data:
        train_datasets = [reg_dataset_classes[ds_name](**common_ds_args, random_sampling=False)
                          for ds_name in args.reg_dataset.split("||")]
    elif args.use_segm_data:
        train_datasets = []
        for ds_name in args.seg_dataset.split('||'):
            seg_dataset_class = seg_dataset_classes.get(ds_name)
            if seg_dataset_class:
                if seg_dataset_class == ReferSegmDataset:
                    all_datasets = args.refer_segm_data.split("||")
                    for d in all_datasets:
                        dataset_class = seg_dataset_class(**common_ds_args, random_sampling=False, refer_segm_data=d)
                        dataset_class._set_len(len(dataset_class.refer_segm_data[d]['images']))
                        train_datasets.append(dataset_class)
                elif seg_dataset_class == SemanticSegmDataset:
                    all_datasets = args.semantic_segm_data.split("||")
                    for d in all_datasets:
                        dataset_class = seg_dataset_class(**common_ds_args, random_sampling=False, refer_segm_data=d)
                        dataset_class._set_len(len(dataset_class.semantic_segm_data[d]['images']))
                        train_datasets.append(dataset_class)
                else:
                    train_datasets.append(seg_dataset_class(**common_ds_args))
    else:
        train_datasets = []

    # Assert that exactly one dataset type is set
    dataset_types_set = sum([args.use_cap_data, args.use_reg_data, args.use_segm_data])
    assert dataset_types_set == 1, "Exactly one dataset type must be set"

    world_size = torch.cuda.device_count()
    # Summing lengths of all datasets
    total_length = sum(len(dataset) for dataset in train_datasets)
    print(f"Training with {total_length} examples.")
    # Calculate steps per epoch
    effective_batch_size = args.batch_size * args.grad_accumulation_steps * world_size
    steps_per_epoch = total_length // effective_batch_size
    # modify steps per epoch
    args.steps_per_epoch = steps_per_epoch

    # Concatenating datasets
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    # Validation datasets
    val_datasets = []
    if not args.no_eval:
        val_dataset_classes = {'CocoCapVal': CocoCapDataset,
                               'RefCOCOgRegVal': RefCocoGRegDataset,
                               'VisGenomeRegVal': VisualGenomeRegDataset,
                               'RefCOCOgSegmVal': ReferSegmDataset,
                               'PsgGCGVal': OpenPsgGCGDataset,
                               'RefCocoGCGVal': RefCOCOgGCGDataset,
                               'FlickrGCGVal': Flickr30kGCGDataset,
                               }
        for val_dataset_name in args.val_dataset.split('|'):
            val_dataset_class = val_dataset_classes.get(val_dataset_name)
            if val_dataset_class:
                if val_dataset_class == ReferSegmDataset:
                    # Modify this if other datasets in refer_segm_data need to be included in val
                    refer_segm_data = 'refcocog'
                    all_datasets = refer_segm_data.split("||")
                    for d in all_datasets:
                        val_dataset_class = val_dataset_class(
                            **common_ds_args, validation=True, refer_segm_data=d, split='val')
                        val_dataset_class._set_len(len(val_dataset_class.refer_segm_data[d]['images']))
                        val_datasets.append(val_dataset_class)
                else:
                    val_datasets.append(val_dataset_class(**common_ds_args, validation=True))

    return train_dataset, val_datasets


def setup_data_loaders(args, train_dataset, val_datasets, tokenizer):
    sampler_args = {"shuffle": True, "drop_last": False}
    train_loader_args = {"batch_size": args.batch_size, "shuffle": False, "num_workers": args.workers,
                         "pin_memory": False}
    val_loader_args = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": args.workers,
                       "pin_memory": False}
    collate_fn_args_train = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=False
    )
    inference_mode = args.mask_validation
    collate_fn_args_val = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=inference_mode
    )

    # Training loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            train_dataset, **sampler_args
            ), collate_fn=collate_fn_args_train, **train_loader_args
        )

    # Validation loader
    val_loader = None
    if val_datasets:
        combined_val_datasets = ConcatDataset(val_datasets)
        val_loader = torch.utils.data.DataLoader(
            combined_val_datasets, **val_loader_args, collate_fn=collate_fn_args_val,
            sampler=torch.utils.data.distributed.DistributedSampler(combined_val_datasets, **sampler_args), )

    return train_loader, val_loader


def initialize_deepspeed(model, tokenizer, args):
    ds_config = {"train_micro_batch_size_per_gpu": args.batch_size,
                 "gradient_accumulation_steps": args.grad_accumulation_steps,
                 "optimizer": {"type": "AdamW", "params": {"lr": args.lr, "weight_decay": 0.0,
                                                           "betas": (args.beta1, args.beta2)}},
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


def resume_training_from_checkpoint(model_engine, args):
    if args.auto_resume and not args.resume:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"Resume training from {args.resume}, start from epoch {args.start_epoch}")


def main(args):
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    prepare_model_for_training(model, tokenizer, args)

    train_dataset, val_datasets = initialize_datasets_and_loaders(args, tokenizer)

    model_engine, optimizer, scheduler = initialize_deepspeed(model, tokenizer, args)
    resume_training_from_checkpoint(model_engine, args)

    train_loader, val_loader = setup_data_loaders(args, train_dataset, val_datasets, tokenizer)
    dataset_iter = iter(train_loader)

    writer = initialize_environment(args)

    if args.eval_only:
        cur_val_loss = validate_model_performance(val_loader, model_engine, 0, writer, args)[0]
        exit()

    epoch_seeds = [random.randint(0, 100000) for _ in range(args.epochs)]

    best_giou, best_ciou, best_val_loss = 0.0, 0.0, np.inf
    for epoch in range(args.start_epoch, args.epochs):
        random.seed(epoch_seeds[epoch])

        dataset_iter = train(train_loader, model_engine, epoch, scheduler, writer, dataset_iter, args)

        if args.mask_validation:
            giou, ciou = validate_model_performance(val_loader, model_engine, epoch, writer, args)
            is_best = giou > best_giou
            best_giou = max(giou, best_giou)
            best_ciou = ciou if is_best else best_ciou
            if args.local_rank == 0:  # Log the progress
                print(f"Epoch: {epoch}, giou: {giou}, ciou: {ciou}, best_giou: {best_giou}, best_ciou: {best_ciou}")
            save_checkpoint(model_engine, args, epoch, 'giou-ciou', f"{giou:.4f}-{ciou:.4f}", is_best)
        else:
            cur_val_loss = validate_model_performance(val_loader, model_engine, epoch, writer, args)
            is_best = cur_val_loss < best_val_loss
            best_val_loss = min(cur_val_loss, best_val_loss)
            if args.local_rank == 0:  # Log the progress
                print(f"Epoch: {epoch}, Current Validation Loss: {cur_val_loss:.4f}, Best Validation Loss: {best_val_loss:}")
            save_checkpoint(model_engine, args, epoch, 'loss', f"{cur_val_loss:.4f}", is_best)


def save_checkpoint(model_engine, args, epoch, metric_name, metric_value, is_best):
    """ Saves the model checkpoint. """
    # If the checkpoint is the best, save it in ckpt_model_best, else in ckpt_model_last_epoch
    save_dir_name = "ckpt_model_best" if is_best else "ckpt_model_last_epoch"
    save_dir = os.path.join(args.log_dir, save_dir_name)
    # Ensure the directory exists
    if args.local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        ckpt_filename = f"epoch_{epoch}_val_{metric_name}_{metric_value}.pth"
        torch.save({"epoch": epoch, f"val_{metric_name}": metric_value}, os.path.join(save_dir, ckpt_filename))
    torch.distributed.barrier()
    model_engine.save_checkpoint(save_dir)


def train(data_loader, model, epoch, scheduler, writer, dataset_iter, args):
    """Main training loop."""

    def get_next_input(iterator, data_loader):
        """Retrieve next input from the iterator, or reinitialize if necessary."""
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iterator = iter(data_loader)
            return next(new_iterator), new_iterator

    def log_progress():
        """Log training progress."""
        if global_step % args.print_freq == 0:
            if args.distributed:
                for tracker in trackers.values():
                    tracker.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                for key, tracker in trackers.items():
                    writer.add_scalar(f"train/{key}", tracker.avg, global_step)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step)

            for tracker in trackers.values():
                tracker.reset()

    batch_time = AverageMeter("Time", ":.4f")
    data_time = AverageMeter("Data", ":.4f")
    trackers = {"loss": AverageMeter("Loss", ":.4f"),
                "ce_loss": AverageMeter("CeLoss", ":.4f"),
                "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f"),
                "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f"),
                "mask_loss": AverageMeter("MaskLoss", ":.4f")}
    progress = ProgressMeter(args.steps_per_epoch, list(trackers.values()), prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for _ in range(args.grad_accumulation_steps):
            # Select data loader based on step choice
            data_batch, new_iter = get_next_input(dataset_iter, data_loader)
            dataset_iter = new_iter

            data_time.update(time.time() - end)
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                if data_batch[key] is not None:
                    data_batch[key] = data_batch[key].bfloat16()

            output_dict = model(**data_batch)

            # Update training metrics
            for key, tracker in trackers.items():
                if key in output_dict:
                    tracker.update(output_dict[key].item(), data_batch["global_enc_images"].size(0))

            model.backward(output_dict["loss"])
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()
        log_progress()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return dataset_iter


def validate_model_performance(validation_loader, training_model, current_epoch, tensorboard_writer, args):
    if args.mask_validation:
        # For use with only segmentation/GCG type datasets
        trackers = {"intersection": AverageMeter("Intersec", ":.4f", Summary.SUM),
                    "union": AverageMeter("Union", ":.4f", Summary.SUM),
                    "gIoU": AverageMeter("gIoU", ":.4f", Summary.SUM)}

        training_model.eval()
        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                data_batch[key] = data_batch[key].bfloat16()
            torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                results = training_model(**data_batch)

            predictions = results["pred_masks"]
            gt_masks = results["gt_masks"][0].int()
            # Note: An error at this line may suggest that the dataset used for validation does not support
            # segmentation tasks. Ensure that the dataset is appropriate for segmentation analysis.
            predicted_masks = (predictions[0] > 0).int()
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

        if args.local_rank == 0:
            tensorboard_writer.add_scalar("val/giou", global_iou, current_epoch)
            tensorboard_writer.add_scalar("val/ciou", class_iou, current_epoch)
            print("giou: {:.4f}, ciou: {:.4f}".format(global_iou, class_iou))

        return global_iou, class_iou
    else:
        # Initializing performance trackers
        trackers = {"loss": AverageMeter("Loss", ":.4f"), "ce_loss": AverageMeter("CeLoss", ":.4f"),
                    "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f"),
                    "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f"),
                    "mask_loss": AverageMeter("MaskLoss", ":.4f")}

        # Prepare model for validation phase
        # Hack to get the loss
        training_model.train()

        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                if data_batch[key] is not None:
                    data_batch[key] = data_batch[key].bfloat16()
            torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                predictions = training_model(**data_batch)
            # Update performance metrics)
            for key, tracker in trackers.items():
                tracker.update(predictions[key].item(), data_batch["global_enc_images"].size(0))

        # Synchronize metrics across processes
        for tracker in trackers.values():
            tracker.all_reduce()
        # Calculate average validation loss
        avg_val_loss = trackers["ce_loss"].avg
        # Tensorboard logging for primary process
        if args.local_rank == 0:
            tensorboard_writer.add_scalar("val/loss", avg_val_loss, current_epoch)

        return avg_val_loss


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
