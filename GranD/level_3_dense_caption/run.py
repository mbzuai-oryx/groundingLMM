import os
import argparse
import subprocess
import time
from multiprocessing import Process


def convert_names(names_list):
    suffix = "_level_2_processed.json"
    suffix_len = len(suffix)
    return [name[:-suffix_len] + ".jpg" for name in names_list if name[-suffix_len:] == suffix]


def launch_vicuna(image_names_txt_path, level_2_dir_path, gpu_id, output_dir_path):
    # Set the environment variable for this specific process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Launch the external script
    subprocess.run(["python", "query_vicuna_vLLM_level_3.py",
                    "--image_names_txt_path", image_names_txt_path,
                    "--level_2_dir_path", level_2_dir_path,
                    "--output_directory_path", output_dir_path])


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images across multiple GPUs.")

    parser.add_argument("--image_dir_path", required=True)
    parser.add_argument("--level_2_dir_path", type=str, required=True,
                        help="Path to the processed level-2 directory.")
    parser.add_argument("--gpu_ids", type=str, required=True, help="Comma-separated GPU IDs.")
    parser.add_argument("--output_dir_path", required=True, help="Path to the output dir")
    parser.add_argument("--job_id", required=True, type=int)

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Convert the comma-separated GPU IDs to a list of integers
    gpu_ids = list(map(int, args.gpu_ids.split(',')))

    image_names = os.listdir(args.image_dir_path)

    # Split the image names across the GPUs
    num_images_per_gpu = len(image_names) // len(gpu_ids)

    processes = []
    temp_files = []  # List to keep track of temporary files created

    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * num_images_per_gpu
        end_idx = start_idx + num_images_per_gpu if i != len(gpu_ids) - 1 else None  # take the rest for the last GPU

        # Write to a temporary file
        timestamp = time.strftime("%Y%m%d%H%M%S")
        txt_path = f"{args.job_id}_{timestamp}_gpu_{gpu_id}.txt"
        temp_files.append(txt_path)
        with open(txt_path, "w") as f:
            for name in image_names[start_idx:end_idx]:
                f.write(name + "\n")

        # Launch a new process for this GPU
        p = Process(target=launch_vicuna, args=(txt_path, args.level_2_dir_path, gpu_id, args.output_dir_path))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Remove all temporary files created
    for temp_file in temp_files:
        os.remove(temp_file)


if __name__ == "__main__":
    main()
