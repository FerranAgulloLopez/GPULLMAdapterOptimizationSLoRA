import argparse
import os
from shutil import copytree


def create_lora(old_lora_dir: str, new_lora_dir: str):
    copytree(old_lora_dir, new_lora_dir)


def main(lora_dir: str, output_path: str, number_replicas: int):
    os.makedirs(output_path, exist_ok=True)
    for lora_index in range(number_replicas):
        new_lora_dir = os.path.join(output_path, f'dummy-lora_{lora_index}')
        create_lora(lora_dir, new_lora_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate dummy LORAs from existent one")
    parser.add_argument("--lora-dir", type=str, help="Existent LORA directory", required=True)
    parser.add_argument("--output-path", type=str, help="Directory to save dummy LORAs", required=True)
    parser.add_argument("--number-replicas", type=int, help="The number of dummy replicas to create", required=True)
    args = parser.parse_args()
    main(args.lora_dir, args.output_path, args.number_replicas)
