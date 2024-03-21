# Download Pretrained Checkpoints

## Tag2Text and RAM
1. Download it from [recognize-anything/tag2text_swin_14m.pth](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/tag2text_swin_14m.pth)
2. Download it from [recognize-anything/ram_swin_large_14m.pth](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth)

## Command to Run

```bash
# Specify the available number of GPUs
export NUM_GPU=8
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=1211 --use_env infer.py --image_dir_path <path to the directory containing images> --output_dir_path <base path to store the predictions> --model-type tag2text --checkpoint <path to the checkpoints - tag2text_swin_14m.pth>

python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=1212 --use_env infer.py --image_dir_path <path to the directory containing images> --output_dir_path <base path to store the predictions> --model-type ram --checkpoint <path to the checkpoints - ram_swin_large_14m.pth>

```
For help, run
```bash
python infer.py -h
```