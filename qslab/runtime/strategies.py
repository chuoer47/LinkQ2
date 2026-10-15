"""Interpret logical checkpoint tensors according to the declared algorithm."""
from abc import ABC, abstractmethod

from qslab.quantization import QuantizationMethod


class RuntimeQuantization(ABC):
    symmetric = True
    tensor_suffixes = ('', '.scale')

    @abstractmethod
    def read_tensors(self, file, name):
        """Read one layer with this algorithm's tensor and transform semantics."""

    def dequantize(self, data):
        scale = data['scale'].float()
        zero = 0 if self.symmetric else data['zero_point'].float().unsqueeze(-1)
        dense = ((data['values'].float().reshape(*scale.shape, -1) - zero)
                 * scale.unsqueeze(-1)).reshape_as(data['values'])
        return dense


class RTNRuntime(RuntimeQuantization):
    def read_tensors(self, file, name):
        return {'values': file.get_tensor(name), 'scale': file.get_tensor(name + '.scale'),
                'zero_point': None, 'input_scale': None}


class RTNClipRuntime(RTNRuntime):
    """Clipping changes offline codes; online execution matches symmetric RTN."""


class AWQRuntime(RTNRuntime):
    tensor_suffixes = RTNRuntime.tensor_suffixes + ('.input_scale',)

    def read_tensors(self, file, name):
        data = super().read_tensors(file, name)
        data['input_scale'] = file.get_tensor(name + '.input_scale')
        return data

    def dequantize(self, data):
        return super().dequantize(data) / data['input_scale'].float()


class AWQAffineRuntime(AWQRuntime):
    symmetric = False
    tensor_suffixes = AWQRuntime.tensor_suffixes + ('.zero_point',)

    def read_tensors(self, file, name):
        data = super().read_tensors(file, name)
        data['zero_point'] = file.get_tensor(name + '.zero_point')
        return data


RUNTIME_STRATEGIES = {
    QuantizationMethod.RTN: RTNRuntime,
    QuantizationMethod.RTN_CLIP: RTNClipRuntime,
    QuantizationMethod.AWQ: AWQRuntime,
    QuantizationMethod.AWQ_AFFINE: AWQAffineRuntime,
}
