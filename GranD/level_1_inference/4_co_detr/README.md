# Setup the Environment
Please follow the instruction at [Sense-X/Co-DETR](https://github.com/Sense-X/Co-DETR?tab=readme-ov-file#install) to setup the environment.

# Download Pretrained Checkpoints
Please use [google drive link](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) to download `co_deformable_detr_swin_large_900q_3x_coco.pth` checkpoints.

## Command to Run

```bash
python launch_codetr_multi_gpu_inference.py --image_dir_path <path to the directory containing images> --output_dir_path <base path to store the predictions> --ckpt_path <Path to the checkpoints - co_deformable_detr_swin_large_900q_3x_coco.pth> --gpu_ids <comma separated gpu ids to run inference on multiple gpus>
```
For help, run
```bash
python launch_codetr_multi_gpu_inference.py -h
python infer.py -h
```