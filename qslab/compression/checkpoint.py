"""Write complete quantized checkpoints using the source shard layout."""
import json
from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def save_checkpoint(artifact, model_path, output):
    artifact.validate()
    settings = {(w.group_size, w.symmetric) for w in artifact.weights.values()}
    if len(settings) != 1:
        raise ValueError("one checkpoint uses one group size and symmetry setting")
    group_size, symmetric = settings.pop()
    source, destination = Path(model_path), Path(output)
    index_path = source / 'model.safetensors.index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = dict(index['weight_map'])
    else:
        with safe_open(str(source / 'model.safetensors'), framework='pt') as file:
            weight_map = {key: 'model.safetensors' for key in file.keys()}
        index = None
    missing = artifact.weights.keys() - weight_map.keys()
    if missing:
        raise ValueError(f"quantized weights absent from source: {sorted(missing)}")
    destination.mkdir(parents=True, exist_ok=False)
    for file in source.iterdir():
        if file.is_file() and file.suffix != '.safetensors' and file.name not in ('config.json', index_path.name):
            shutil.copy2(file, destination / file.name)
    config = json.loads((source / 'config.json').read_text())
    config['quantization_config'] = {'quant_method': artifact.algorithm.value,
                                    'bits': 4, 'group_size': group_size, 'symmetric': symmetric}
    layers = [name.removesuffix('.weight') for name in artifact.weights]
    (destination / 'quantized_layers.json').write_text(json.dumps(layers, indent=2) + '\n')
    total_size = 0
    for shard in dict.fromkeys(weight_map.values()):
        tensors = {}
        with safe_open(str(source / shard), framework='pt', device='cpu') as file:
            for name in file.keys():
                if name not in artifact.weights:
                    tensors[name] = file.get_tensor(name)
                    continue
                quantized = artifact.weights[name]
                tensors[name] = quantized.values.to(device='cpu', dtype=torch.int8).contiguous()
                tensors[name + '.scale'] = quantized.scale.cpu().contiguous()
                if not symmetric:
                    tensors[name + '.zero_point'] = quantized.zero_point.cpu().contiguous()
                input_scale = artifact.transforms.get('input_scale', {}).get(name)
                if input_scale is not None:
                    tensors[name + '.input_scale'] = input_scale.cpu().contiguous()
        total_size += sum(t.numel() * t.element_size() for t in tensors.values())
        save_file(tensors, str(destination / shard), metadata={'format': 'pt'})
        weight_map.update({name: shard for name in tensors})
        del tensors
        print(f'saved {shard}', flush=True)
    if index is not None:
        index['weight_map'] = weight_map
        index.setdefault('metadata', {})['total_size'] = total_size
        (destination / index_path.name).write_text(json.dumps(index, indent=2) + '\n')
    (destination / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
