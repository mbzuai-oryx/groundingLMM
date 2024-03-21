# Setup the Environment

```bash
pip install -r requirements.txt
```

# Download Pretrained Checkpoints

Download checkpoints from the following links and place it in the current directory.
```bash
wget -c https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth
```

## Command to Run

```bash
# Specify the available number of GPUs
export NUM_GPU=8
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=1211 --use_env infer.py --image_dir_path <path to the directory containing images> --output_dir_path <base path to store the predictions>
```

For help, run

```bash
python infer.py -h
```
