import os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
import torch
import json
import subprocess


class CustomLevel1JsonDataset(Dataset):
    def __init__(self, image_dir_path, level_2_dir_path, output_dir_path,
                 image_transform):
        self.image_names = []
        self.image_paths = []
        self.lmdb_keys = []
        self.output_dir_path = output_dir_path

        all_image_names = os.listdir(image_dir_path)
        self.level_2_dir_path = level_2_dir_path

        for image_name in all_image_names:
            self.image_names.append(image_name)
            self.image_paths.append(f"{image_dir_path}/{image_name}")

        self.transform = image_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(self.image_paths[idx])
        image_width, image_height = image.size

        json_contents = json.load(open(f"{self.level_2_dir_path}/{image_name[:-4]}.json", 'r'))

        box_crops, labels = {}, {}
        output_json_file_path = f"{self.output_dir_path}/{self.image_names[idx][:-4]}.json"
        if os.path.exists(output_json_file_path):
            return self.image_names[idx], box_crops, labels
        # box_crops: {id1: crop1, id2: crop2, id3: crop3, ...}
        # labels: {id1: [labels1], id2: [labels2], id3: [labels3], ...}
        for obj_category in ['objects', 'floating_objects']:
            for obj in json_contents[image_name][obj_category]:
                if len(obj['labels']) == 1:
                    continue
                obj_id = obj['id']

                bbox = obj['bbox']
                bbox = [max(coord, 0) for coord in bbox]  # Check for negative coordinates and correct them
                # Ensure bounding box is bounded by image dimensions
                bbox[2] = min(bbox[2], image_width)  # x_max shouldn't exceed image width
                bbox[3] = min(bbox[3], image_height)  # y_max shouldn't exceed image height
                # Optional: Check if the corrected bbox is still valid
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue  # skip this bounding box after corrections

                crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                crop_w, crop_h = crop.size
                if crop_w == 0 or crop_h == 0:
                    continue
                if self.transform:
                    # try:
                    crop = self.transform(crop)
                    # except ZeroDivisionError:
                    #     print(f"Error with image: {self.image_paths[idx]} and bbox: {bbox}")
                    #     raise
                box_crops[obj_id] = crop
                labels[obj_id] = obj['labels']

        return image_name, box_crops, labels


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
