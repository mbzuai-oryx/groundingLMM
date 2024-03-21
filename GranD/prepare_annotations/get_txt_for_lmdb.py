import lmdb
import argparse


def save_keys_to_file(db_path, output_file):
    env = lmdb.open(db_path, readonly=True)

    with env.begin() as txn:
        cursor = txn.cursor()

        with open(output_file, 'w') as f:
            for key, _ in cursor:
                f.write(key.decode('utf-8') + '\n')

    env.close()


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Save all keys from an LMDB database to a text file.')
    parser.add_argument('--db_path', required=True, help='Path to the LMDB database.')
    parser.add_argument('--output_file', required=True, help='Path to the output text file.')

    # Parse the command line arguments
    args = parser.parse_args()

    # Save all keys in the LMDB database to the output file
    save_keys_to_file(args.db_path, args.output_file)