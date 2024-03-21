import os
import lmdb
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import concurrent


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create an LMDB database from JSON files.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing JSON files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store the LMDB database.")

    return parser.parse_args()


def process_file(args):
    file, input_dir = args
    try:
        file_path = os.path.join(input_dir, file)
        with open(file_path) as f:
            data = json.load(f)
        image_name = list(data.keys())[0]
        return (image_name.encode('utf-8'), json.dumps(data).encode('utf-8'))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Exception processing: {file}, Error: {str(e)}")
        return None


def chunked_iterable(iterable, chunk_size):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]


def create_lmdb(input_dir, output_dir):
    env = lmdb.open(output_dir, map_size=1099511627776 * 2, readonly=False, meminit=False, map_async=True)
    all_files = os.listdir(input_dir)

    with env.begin(write=True) as txn:
        with ProcessPoolExecutor(max_workers=16) as executor:
            for chunk in tqdm(chunked_iterable(all_files, 10000), total=int(len(all_files)/10000)):  # Adjust chunk size as needed
                futures = [executor.submit(process_file, (file, input_dir)) for file in chunk]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        txn.put(*result)

    env.sync()
    env.close()


if __name__ == "__main__":
    args = parse_arguments()
    create_lmdb(args.input_dir, args.output_dir)