from chainer import function, cuda
from chainer.utils import type_check, force_array
import chainer.variable


class VecSubMat(function.Function):

    def __init__(self, lhs_bwd):
        self.lhs_bwd = lhs_bwd

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == in_types[1].dtype,
            # check if the mini_batch sizes are same
            in_types[0].shape[0] == in_types[1].shape[0]
        )

    def forward_cpu(self, x):
        return force_array(x[0] - x[1]),

    def forward_gpu(self, x):
        y = x[1].copy()
        cuda.elementwise(
            'float* y, float* b, int nc',
            'y[i] = b[i / nc] - y[i];',
            'sub_bias')(y, x[0], x[1].shape[1])
        return y,

    def backward_cpu(self, x, gy):
        glhs = None
        if self.lhs_bwd:
            s = x[0].shape
            glhs = gy[0].sum(1).reshape(s) / s[1]
        return glhs, -gy[0]

    def backward_gpu(self, x, gy):
        glhs = None
        if self.lhs_bwd:
            with cuda.using_cumisc():
                s = x[0].shape
                glhs = cuda.cumisc.sum(gy[0], 1).reshape(s) / s[1]
        return glhs, -gy[0]


def vec_sub_mat(lhs, rhs, lhs_bwd=True):  # lhs - rhs
    assert isinstance(rhs, chainer.variable.Variable)
    return VecSubMat(lhs_bwd)(lhs, rhs)