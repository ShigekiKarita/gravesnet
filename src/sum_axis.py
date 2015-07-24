from chainer import function, cuda
from chainer.utils import type_check
import numpy


class SumAxis(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1
        )

    def check_type_backward(self, in_types, grad_types):
        type_check.expect(
            in_types[0].shape[0] == grad_types[0].shape[0]
        )

    def forward_cpu(self, inputs):
        return inputs[0].sum(1).reshape((inputs[0].shape[0], 1)),

    def forward_gpu(self, inputs):
        with cuda.using_cumisc():
            return cuda.cumisc.sum(inputs[0], 1).reshape((inputs[0].shape[0], 1)),

    def backward_cpu(self, inputs, grad_outputs):
        return numpy.full_like(inputs[0], grad_outputs[0]),

    def backward_gpu(self, inputs, grad_outputs):
        gx = cuda.empty_like(inputs[0])
        cuda.elementwise(
            'float* y, const float* b, const int n_channel',
            'y[i] = b[i % n_channel]',
            'sum_axis_bwd')(gx, grad_outputs[0], grad_outputs[0].size)
        return gx,


def sum_axis(x):
    return SumAxis()(x)