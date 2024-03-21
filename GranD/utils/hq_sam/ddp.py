import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
import torch
import json
import subprocess


class CustomJsonDataset(Dataset):
    def __init__(self, image_dir_path, level_3_processed_path, output_dir_path):
        self.image_names = []
        self.image_paths = []
        self.json_paths = []
        self.output_dir_path = output_dir_path

        all_image_names = os.listdir(image_dir_path)

        for image_name in all_image_names:
            self.image_names.append(image_name)
            self.image_paths.append(f"{image_dir_path}/{image_name}")
            self.json_paths.append(f"{level_3_processed_path}/{image_name[:-4]}.json")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        boxes = {}
        output_json_file_path = f"{self.output_dir_path}/{self.image_names[idx][:-4]}.json"
        if os.path.exists(output_json_file_path):
            return self.image_paths[idx], self.image_names[idx], {}, boxes

        with open(self.json_paths[idx], 'r') as f:
            json_contents = json.load(f)

        image_name = self.image_names[idx]
        # Get all valid object & floating object ids
        valid_object_ids = set()
        for caption in json_contents[image_name]["short_captions"]:
            details = caption["details"]
            for grounding in details:
                valid_object_ids.add(grounding["id"])
        dense_caption = json_contents[image_name]["dense_caption"]
        details = dense_caption["details"]
        for grounding in details:
            ids = grounding["ids"]
            for id in ids:
                valid_object_ids.add(id)

        # Masks for all objects
        for obj in json_contents[image_name]['objects']:
            if "segmentation" not in obj.keys():
                obj_id = obj['id']
                boxes[obj_id] = obj['bbox']
        # Masks for floating objects included in grounding
        for obj in json_contents[image_name]['floating_objects']:
            # if ("segmentation" not in obj.keys()) and (obj['id'] in valid_object_ids):
            if "segmentation" not in obj.keys():
                obj_id = obj['id']
                boxes[obj_id] = obj['bbox']

        return self.image_paths[idx], image_name, json_contents, boxes


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
