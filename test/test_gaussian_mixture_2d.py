import unittest

import numpy
import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr, condition

from src.gaussian_mixture_2d import gaussian_mixture_2d
from src import gravesnet


if cuda.available:
    cuda.init()


class TestGaussianMixture2d(unittest.TestCase):

    def setUp(self):
        # each 2D-Gaussian contains 6 params: weight, mean(2), stddev(2), corr
        self.ngauss = 5
        input_size = 6 * self.ngauss + 1
        mini_batch = 3
        self.x = numpy.random.randn(mini_batch, input_size).astype(numpy.float32)
        self.t = numpy.random.randn(mini_batch, 2 + 1).astype(numpy.float32)

    def check(self):
        x = chainer.Variable(self.context(self.x))
        t = chainer.Variable(self.context(self.t))
        y = gravesnet.loss_func(self.ngauss, x, t)
        y.creator.forward((x.data,))

    def test_gaussian_mixture_2d_cpu(self):
        self.context = lambda x: x
        self.check()

    @attr.gpu
    def test_gaussian_mixture_2d_gpu(self):
        self.context = cuda.to_gpu
        self.check()



