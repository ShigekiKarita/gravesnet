from unittest import TestCase

import chainer
from chainer import cuda
from chainer.functions import concat, split_axis
from chainer.gradient_check import assert_allclose
from chainer import testing
from chainer.testing import attr, condition

import numpy

from src.spilit_axis import split_axis_by_widths


if cuda.available:
    cuda.init()


class TestSplitAxis(TestCase):

    def setUp(self):
        # self.ws = [2, 4]
        h = numpy.random.randint(1, 5)
        self.ws = [numpy.random.randint(1, 5)] * h * numpy.random.randint(1, 5)
        self.ws[0] = h
        self.mini_batch = numpy.random.randint(1, 5)  # FIXME: set 1 -> FAIL

    def check(self, widths):
        x_size = sum(self.ws)
        x = numpy.arange(self.mini_batch * x_size,
                         dtype=numpy.float32).reshape(self.mini_batch, x_size)
        x = chainer.Variable(self.context(x))
        y = split_axis_by_widths(x, widths)
        z = concat(y)
        assert_allclose(x.data, z.data)

    @condition.retry(100)
    def test_split_axis_cpu(self):
        self.context = lambda x: x
        self.check(self.ws)
        self.check(self.ws[0])

    @condition.retry(100)
    @attr.gpu
    def test_split_axis_gpu(self):
        self.context = cuda.to_gpu
        self.check(self.ws)
        self.check(self.ws[0])


testing.run_module(__name__, __file__)