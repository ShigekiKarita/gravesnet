import unittest

import chainer
from chainer import testing
from chainer import cuda
from chainer import gradient_check
from chainer.testing import attr, condition
import numpy

from src.functions.sqrt import mysqrt

if cuda.available:
    cuda.init()



class UnaryFunctionsTestBase(object):

    def make_data(self):
        raise NotImplementedError()

    def setUp(self):
        self.x, self.gy = self.make_data()

    def check_forward(self, op, op_np, x_data):
        x = chainer.Variable(x_data)
        y = op(x)
        gradient_check.assert_allclose(
            op_np(self.x), y.data, atol=1e-7, rtol=1e-7)

    def forward_cpu(self, op, op_np):
        self.check_forward(op, op_np, self.x)

    def forward_gpu(self, op, op_np):
        self.check_forward(op, op_np, cuda.to_gpu(self.x))

    def check_backward(self, op, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = op(x)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = gradient_check.numerical_grad(f, (x.data,), (y.grad,))

        gradient_check.assert_allclose(gx, x.grad)

    def backward_cpu(self, op):
        self.check_backward(op, self.x, self.gy)

    def backward_gpu(self, op):
        self.check_backward(op, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @condition.retry(3)
    def test_sqrt_forward_cpu(self):
        self.forward_cpu(mysqrt, numpy.sqrt)

    @condition.retry(3)
    @attr.gpu
    def test_sqrt_forward_gpu(self):
        self.forward_cpu(mysqrt, numpy.sqrt)

    @condition.retry(3)
    def test_sqrt_backward_cpu(self):
        self.backward_cpu(mysqrt)

    @attr.gpu
    @condition.retry(3)
    def test_sqrt_backward_gpu(self):
        self.backward_gpu(mysqrt)


class TestUnaryFunctionsSimple(UnaryFunctionsTestBase, unittest.TestCase):
    def make_data(self):
        x = numpy.random.uniform(.5, 1, (3, 2)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        return x, gy


class TestUnaryFunctionsZeroDimension(UnaryFunctionsTestBase,
                                      unittest.TestCase):
    def make_data(self):
        x = numpy.random.uniform(.5, 1, ()).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
        return x, gy


testing.run_module(__name__, __file__)