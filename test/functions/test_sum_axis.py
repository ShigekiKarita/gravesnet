from unittest import TestCase

from chainer import testing
import chainer
from chainer import cuda
from chainer.gradient_check import assert_allclose
from chainer.testing import attr, condition
import numpy

from src.functions.sum_axis import sum_axis

if cuda.available:
    cuda.init()


class TestSumAxis(TestCase):

    def setUp(self):
        r = numpy.random.randint(2, 5)
        c = numpy.random.randint(2, 5)
        self.w = numpy.arange(r).reshape(r, 1).astype(numpy.float32)
        self.x = numpy.full((r, c), self.w).astype(numpy.float32)
        self.y = c * self.w

    def check(self, context):
        x = chainer.Variable(context(self.x))
        y = sum_axis(x)
        assert_allclose(y.data, context(self.y))

        y.grad = context(self.w)
        y.backward()
        assert_allclose(x.grad, context(self.x))

    @condition.retry(100)
    def test_cpu(self):
        self.check(lambda x: x)

    @condition.retry(100)
    @attr.gpu
    def test_gpu(self):
        self.check(cuda.to_gpu)


testing.run_module(__name__, __file__)