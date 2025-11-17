import argparse
import json
import os
from typing import List

import safetensors.torch
import torch


def create_lora(old_lora_dir: str, new_lora_dir: str, new_rank: int, new_layers: int, simplified: bool):
    # Load old adapter
    lora_config_path = os.path.join(old_lora_dir, "adapter_config.json")
    lora_tensor_path = os.path.join(old_lora_dir, "adapter_model.safetensors")
    lora_bin_file_path = os.path.join(old_lora_dir, "adapter_model.bin")
    new_embeddings_tensor_path = os.path.join(old_lora_dir, "new_embeddings.safetensors")
    new_embeddings_bin_file_path = os.path.join(old_lora_dir, "new_embeddings.bin")

    if os.path.isfile(lora_tensor_path):
        tensors = safetensors.torch.load_file(lora_tensor_path)
    elif os.path.isfile(lora_bin_file_path):
        raise NotImplemented(f"bin format for tensors not implemented")
    else:
        raise ValueError(f"{old_lora_dir} doesn't contain tensors")

    if os.path.isfile(new_embeddings_tensor_path):
        embeddings = safetensors.torch.load_file(new_embeddings_tensor_path)
    elif os.path.isfile(new_embeddings_bin_file_path):
        raise NotImplemented(f"bin format for embeddings not implemented")
    else:
        raise NotImplemented(f"no implementation for not embeddings option")

    with open(lora_config_path) as f:
        config = json.load(f)
    rank = config["r"]

    print(f'Original rank: {rank}')

    # Check old adapter
    lora_a_count = 0
    lora_b_count = 0
    layers_ids = set()
    for tensor_name, tensor_array in tensors.items():
        if '_A' in tensor_name:
            lora_a_count += 1
        elif '_B' in tensor_name:
            lora_b_count += 1
        else:
            raise Exception(f'Unrecognized tensor name {tensor_name}')

        if len(tensor_array.shape) != 2:
            raise Exception(f'Unrecognized tensor array shape {tensor_array.shape} from {tensor_name}')

        if '_A' in tensor_name and tensor_array.shape[0] != rank:
            raise Exception(f'Unrecognized rank from {tensor_name}')
        elif '_B' in tensor_name and tensor_array.shape[1] != rank:
            raise Exception(f'Unrecognized rank from {tensor_name}')

        if '.layers.' in tensor_name:
            layer_id: int = int(tensor_name.split('.')[4])
            layers_ids.add(layer_id)

    print(f'Number of LORA tensors: {len(tensors)}. A -> {lora_a_count}. B -> {lora_b_count}')
    print(f'Number of embeddings: {len(embeddings)}')
    print(f'Number of layers: {len(layers_ids)}')

    # Create new adapter
    config['r'] = new_rank
    if simplified:
        config['target_modules'] = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
    new_tensors = {}

    if new_layers is None:
        new_layers = len(layers_ids)
    layer_example_id: int = next(iter(layers_ids))
    layer_example_tensors = {}

    for tensor_name, tensor_array in tensors.items():  # Not layer weights
        if '.layers.' not in tensor_name:
            if simplified:
                continue
            if '_A' in tensor_name:
                new_tensors[tensor_name] = torch.zeros((new_rank, tensor_array.shape[1]), dtype=tensor_array.dtype)
            else:
                new_tensors[tensor_name] = torch.zeros((tensor_array.shape[0], new_rank), dtype=tensor_array.dtype)
        elif f'.layers.{layer_example_id}' in tensor_name:
            layer_example_tensors[tensor_name] = tensor_array

    for new_layer_id in range(new_layers):  # Layer weights
        for tensor_name, tensor_array in layer_example_tensors.items():
            new_tensor_name: str = tensor_name.replace(f'.layers.{layer_example_id}', f'.layers.{new_layer_id}')
            if simplified and 'self_attn' not in tensor_name:
                continue
            if '_A' in new_tensor_name:
                new_tensors[new_tensor_name] = torch.zeros((new_rank, tensor_array.shape[1]), dtype=tensor_array.dtype)
            else:
                new_tensors[new_tensor_name] = torch.zeros((tensor_array.shape[0], new_rank), dtype=tensor_array.dtype)

    # save new adapter
    with open(os.path.join(new_lora_dir, "adapter_config.json"), mode='w') as f:
        json.dump(config, f)
    safetensors.torch.save_file(new_tensors, os.path.join(new_lora_dir, "adapter_model.safetensors"))
    safetensors.torch.save_file(embeddings, os.path.join(old_lora_dir, "new_embeddings.safetensors"))

    print(f'Created new LORA adapter: {new_lora_dir}')


def main(lora_dir: str, output_path: str, ranks: List[int], layers: int, simplified: bool):
    os.makedirs(output_path, exist_ok=True)
    lora_name = os.path.basename(os.path.normpath(lora_dir))
    for rank in ranks:
        new_lora_dir = f'{lora_name}_dummy_rank_{str(rank)}'
        if simplified:
            new_lora_dir += '_simplified'
        new_lora_dir = os.path.join(output_path, new_lora_dir)
        if layers is not None:
            new_lora_dir += f'_layers_{layers}'
        os.makedirs(new_lora_dir)
        create_lora(lora_dir, new_lora_dir, rank, layers, simplified)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creation dummy LORAs from existent one")
    parser.add_argument("--lora-dir", type=str, help="Existent LORA directory", required=True)
    parser.add_argument("--output-path", type=str, help="Directory to save dummy LORAs", required=True)
    parser.add_argument("--ranks", type=int, nargs='+', help="The ranks of the LORAs to create", required=True)
    parser.add_argument("--layers", type=int, help="The ranks of the LORAs to create", required=False, default=None)
    parser.add_argument('--simplified', action='store_true', help='Only copy attention layers', default=False)
    args = parser.parse_args()
    main(args.lora_dir, args.output_path, args.ranks, args.layers, args.simplified)
