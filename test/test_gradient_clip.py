import unittest

import chainer
from chainer import cuda
from chainer.gradient_check import assert_allclose
from chainer.testing import attr, condition, run_module

import numpy

from src.gradient_clip import gradient_clip


if cuda.available:
    cuda.init()


class TestGradientClip(unittest.TestCase):

    def setUp(self):
        self.a = numpy.random.random()
        self.b = numpy.random.random()
        self.x = numpy.random.randn(4, 3).astype(numpy.float32)
        self.y_grad = numpy.random.randn(4, 3).astype(numpy.float32)
        lower = min(self.a, self.b)
        upper = max(self.a, self.b)
        self.x_grad_ab = numpy.clip(self.y_grad, lower, upper)
        self.x_grad_a = numpy.clip(self.y_grad, -abs(self.a), abs(self.a))

    def check_backward(self, f):
        x = chainer.Variable(f(self.x))

        y = gradient_clip(x, self.a, self.b)
        y.creator.forward((x.data,))
        y.grad = f(self.y_grad)
        y.backward()
        assert_allclose(y.data, x.data)
        assert_allclose(x.grad, f(self.x_grad_ab))

        y = gradient_clip(x, self.a)
        y.creator.forward((x.data,))
        y.grad = f(self.y_grad)
        y.backward()
        assert_allclose(y.data, x.data)
        assert_allclose(x.grad, f(self.x_grad_a))


    @condition.retry(100)
    def test_backward_cpu(self):
        self.check_backward(lambda x: x)

    @attr.gpu
    @condition.retry(100)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu)


run_module(__name__, __file__)