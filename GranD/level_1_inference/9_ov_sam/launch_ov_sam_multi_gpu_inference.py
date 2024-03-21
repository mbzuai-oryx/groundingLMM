import os
import argparse
import subprocess
import time
from multiprocessing import Process


def launch_ovsam(image_names_txt_path, gpu_id, image_dir_path, sam_annotations_dir, output_dir_path):
    """Invoke the predict_co-detr.py script."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    subprocess.run([
        "python",
        "infer.py",
        "--image_names_txt_path", image_names_txt_path,
        "--image_dir_path", image_dir_path,
        "--sam_annotations_dir", sam_annotations_dir,
        "--output_dir_path", output_dir_path,
        "--sam_annotations_dir", output_dir_path,
    ])


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run co-detr prediction across multiple GPUs.")

    parser.add_argument("--image_dir_path", type=str, required=True, help="Path to the image directory.")
    parser.add_argument("--output_dir_path", type=str, required=True, help="Path for saving the outputs.")
    parser.add_argument("--sam_annotations_dir", required=True,
                        help="path to the directory containing all sam annotations.")
    parser.add_argument("--gpu_ids", type=str, required=True, help="Comma-separated GPU IDs.")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Convert the comma-separated GPU IDs to a list of integers
    gpu_ids = list(map(int, args.gpu_ids.split(',')))

    # Get the image names from the directory
    # with open(args.image_names_txt_path, 'r') as f:
    #     image_names = [line.strip() for line in f if line.strip()]
    image_names = os.listdir(args.image_dir_path)

    # Split the image names across the GPUs
    num_images_per_gpu = len(image_names) // len(gpu_ids)

    processes = []
    temp_files = []  # List to track temporary files created

    for i, gpu_id in enumerate(gpu_ids):
        time.sleep(1)
        start_idx = i * num_images_per_gpu
        end_idx = start_idx + num_images_per_gpu if i != len(gpu_ids) - 1 else None  # Take the rest for the last GPU

        # Write to a temporary file
        timestamp = time.strftime("%Y%m%d%H%M%S")
        txt_path = f"temp_gpu_{gpu_id}_{timestamp}.txt"
        temp_files.append(txt_path)
        with open(txt_path, "w") as f:
            for name in image_names[start_idx:end_idx]:
                f.write(name + "\n")

        # Launch a new process for this GPU
        p = Process(target=launch_ovsam,
                    args=(txt_path, gpu_id, args.image_dir_path, args.sam_annotations_dir, args.output_dir_path))
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
