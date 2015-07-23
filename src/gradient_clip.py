from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy


class GradientClip(function.Function):
    def __init__(self, lower, upper):
        self.upper = max(lower, upper)
        self.lower = min(lower, upper)

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype == numpy.float32
        )

    def forward(self, inputs):
        return inputs[0],

    def backward_cpu(self, inputs, grad_outputs):
        return numpy.clip(grad_outputs[0], self.lower, self.upper),

    def backward_gpu(self, inputs, grad_outputs):
        return cuda.gpuarray.minimum(cuda.gpuarray.maximum(grad_outputs[0], self.lower), self.upper),


def gradient_clip(x, a, b=None):
    if b is None:
        b = -a
    return GradientClip(a, b)(x)
