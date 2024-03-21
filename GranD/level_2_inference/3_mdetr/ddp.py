import json
import os
from torch.utils.data import Dataset
import torch
import subprocess
import numpy as np
from PIL import Image, ImageFile
import lmdb
from utils import get_llava_phrases, get_blip2_phrases

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomImageDataset(Dataset):
    def __init__(self, image_dir_path, blip2_path, llava_path, transforms):
        self.image_names = os.listdir(image_dir_path)
        self.image_paths = []
        for image_name in self.image_names:
            self.image_paths.append(f"{image_dir_path}/{image_name}")
        self.transforms = transforms
        # Open the BLIP2 and LLaVA LMDBs
        self.blip2_path = blip2_path
        self.llava_path = llava_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image_np = np.array(image)
        if image_np.ndim == 2:
            image_np = np.repeat(image_np[:, :, np.newaxis], 3, axis=2)
            image = Image.fromarray(image_np)
        image_size = image.size
        if self.transforms:
            image = self.transforms(image)

        # Read BLIP2 and LLaVA captions
        blip2_caption = json.load(open(f"{self.blip2_path}/{self.image_names[idx][:-4]}.json", 'r'))
        blip2_caption = blip2_caption[self.image_names[idx]]['blip2']

        llava_caption = json.load(open(f"{self.llava_path}/{self.image_names[idx][:-4]}.json", 'r'))
        llava_caption = llava_caption[self.image_names[idx]]['llava']

        # Extract phrases
        blip2_nouns, blip2_phrases = get_llava_phrases(blip2_caption)
        if not blip2_phrases:
            blip2_nouns, blip2_phrases = get_blip2_phrases(blip2_caption)
        llava_nouns, llava_phrases = get_llava_phrases(llava_caption)
        if not llava_phrases:
            llava_nouns, llava_phrases = get_blip2_phrases(llava_caption)
        phrases = blip2_phrases + llava_phrases
        nouns = blip2_nouns + llava_nouns

        return self.image_names[idx], image, image_size, [blip2_caption, llava_caption], nouns, phrases

    def close(self):
        self.blip2_env.close()
        self.llava_env.close()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
        print('Using distributed mode: 1')
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '3460')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
        print('Using distributed mode: slurm')
        print(f"world: {os.environ['WORLD_SIZE']}, rank:{os.environ['RANK']},"
              f" local_rank{os.environ['LOCAL_RANK']}, local_size{os.environ['LOCAL_SIZE']}")
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
