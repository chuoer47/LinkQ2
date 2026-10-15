"""Load a complete quantized checkpoint without importing offline algorithms."""
import json
from pathlib import Path

from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
import torch

from .backends.quantization import make_quant_linear


class QuantizedWeights:
    def __init__(self, path):
        self.path = Path(path)
        self.config = json.loads((self.path / 'config.json').read_text())['quantization_config']
        if self.config['bits'] != 4:
            raise ValueError('only W4 checkpoints are supported')
        index = self.path / 'model.safetensors.index.json'
        if index.exists():
            self.weight_map = json.loads(index.read_text())['weight_map']
        else:
            with safe_open(str(self.path / 'model.safetensors'), framework='pt') as file:
                self.weight_map = {name: 'model.safetensors' for name in file.keys()}
        self.names = [name.removesuffix('.scale') for name in self.weight_map if name.endswith('.weight.scale')]

    def layer(self, name):
        with safe_open(str(self.path / self.weight_map[name]), framework='pt', device='cpu') as file:
            return {'values': file.get_tensor(name), 'scale': file.get_tensor(name + '.scale'),
                    'zero_point': None if self.config['symmetric'] else file.get_tensor(name + '.zero_point'),
                    'input_scale': file.get_tensor(name + '.input_scale')
                        if name + '.input_scale' in self.weight_map else None}


@torch.no_grad()
def load_checkpoint_weights(model, path, *, backend='gemlite', device='cuda:0', dtype=torch.float16, fake=False):
    weights = QuantizedWeights(path)
    for key in weights.names:
        name = key.removesuffix('.weight')
        old = model.get_submodule(name)
        data = weights.layer(key)
        if tuple(old.weight.shape) != tuple(data['values'].shape):
            raise ValueError(f'weight shape differs: {key}')
        if fake:
            scale = data['scale'].float()
            zero = 0 if data['zero_point'] is None else data['zero_point'].float().unsqueeze(-1)
            dense = ((data['values'].float().reshape(*scale.shape, -1) - zero) * scale.unsqueeze(-1)).reshape_as(data['values'])
            if data['input_scale'] is not None:
                dense /= data['input_scale'].float()
            set_module_tensor_to_device(model, key, device, value=dense, dtype=dtype)
            del dense
        else:
            if old.bias is not None:
                raise ValueError(f'biased quantized projections are not supported: {name}')
            old.weight = None
            layer = make_quant_linear(backend, **data, group_size=weights.config['group_size'],
                symmetric=weights.config['symmetric'], device=device, dtype=dtype)
            parent, _, child = name.rpartition('.')
            setattr(model.get_submodule(parent) if parent else model, child, layer)
        del data
    auxiliary = {name + suffix for name in weights.names for suffix in ('', '.scale', '.zero_point', '.input_scale')}
    for shard in dict.fromkeys(weights.weight_map.values()):
        with safe_open(str(weights.path / shard), framework='pt', device='cpu') as file:
            for name in file.keys():
                if name not in auxiliary:
                    set_module_tensor_to_device(model, name, device, value=file.get_tensor(name), dtype=dtype)
    return len(weights.names)
