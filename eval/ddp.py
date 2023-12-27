import os
import torch
import subprocess
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from eval.utils import bbox_to_x1y1x2y2


class RegionCapDDP(Dataset):
    def __init__(self, annotation_file):
        self.coco = COCO(annotation_file)
        self.image_dict = self.coco.imgs
        self.ann_dict = self.coco.anns
        self.image_dict_keys = list(self.image_dict.keys())


    def __len__(self):
        return len(self.image_dict_keys)

    def __getitem__(self, idx):
        image_id = self.image_dict_keys[idx]
        filename = self.image_dict[image_id]['file_name']
        bbox = bbox_to_x1y1x2y2(self.ann_dict[image_id]['bbox'])
        gt = self.ann_dict[image_id]['caption']

        return image_id, filename, bbox, gt


class GCGEvalDDP(Dataset):
    def __init__(self, image_dir_path):
        self.image_dir_path = image_dir_path
        self.image_ids = os.listdir(image_dir_path)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = f"{self.image_dir_path}/{image_id}"

        return image_id, image_path


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
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
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
