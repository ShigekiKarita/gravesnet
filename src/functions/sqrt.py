from chainer import function, cuda
from chainer.utils import type_check, force_array
import numpy


class Sqrt(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward_cpu(self, inputs):
        self.y = force_array(numpy.sqrt(inputs[0]))
        return self.y,

    def forward_gpu(self, inputs):
        self.y = cuda.cumath.sqrt(inputs[0])
        return self.y,

    def backward_cpu(self, inputs, grad_outputs):
        return force_array(numpy.float32(0.5) * grad_outputs[0] / self.y),

    def backward_gpu(self, inputs, grad_outputs):
        return 0.5 * grad_outputs[0] / self.y,


def mysqrt(x):
    return Sqrt()(x)


