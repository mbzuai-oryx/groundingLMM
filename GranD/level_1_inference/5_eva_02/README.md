# Setup the Environment

Please follow the instruction
at [baaivision/EVA/tree/master/EVA-02/det](https://github.com/baaivision/EVA/tree/master/EVA-02/det#setup) to setup the
environment.

# Download Pretrained Checkpoints

Use the following links to download the model checkpoints and place it in the current directory.

1. [eva02_L_lvis_sys.pth](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_lvis_sys.pth)
2. [eva02_L_lvis_sys_o365.pth](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_lvis_sys_o365.pth)

## Command to Run

```bash
# Specify the available number of GPUs
export NUM_GPU=8
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=1211 --use_env infer.py --image_dir_path <path to the directory containing images> --output_dir_path <base path to store the predictions> --model_name <either eva-02-01 or eva-02-02>
```

For help, run

```bash
python infer.py -h
```