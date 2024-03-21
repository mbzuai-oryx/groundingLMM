# Setup the Environment

For installation, please follow the instructions at [amazon-science/prompt-pretraining/tree/main/third_party/Detic](https://github.com/amazon-science/prompt-pretraining/tree/main/third_party/Detic).

# Download Pretrained Checkpoints

Download checkpoints from the following links and place it in the current directory.
1. [vit_b16_ep20_randaug2_unc1000_16shots_nctx16_cscFalse_ctpend_seed42.pth.tar](https://drive.google.com/file/d/1C8oU6cWkJdU3Q3IHaqTcbIToRLo9bMnu/view?usp=sharing)
2. [Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size_pomp.pth](https://drive.google.com/file/d/1TwrjcUYimkI_f9z9UZXCmLztdgv31Peu/view?usp=sharing)

## Command to Run

```bash
python launch_pomp_multi_gpu_inference.py --image_dir_path <path to the directory containing images> --output_dir_path <base path to store the predictions> --tags_dir_path <path containing tag2text and ram predictions> --gpu_ids <comma separated gpu ids to run inference on multiple gpus>
```

For help, run

```bash
python infer.py -h
```
