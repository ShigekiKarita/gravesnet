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
        self.ngauss = 3
        input_size = 6 * self.ngauss + 1
        mini_batch = 1
        self.x = numpy.random.uniform(-1, 1, (mini_batch, input_size)).astype(numpy.float32)
        self.t_x = numpy.random.uniform(-1, 1, (mini_batch, 2)).astype(numpy.float32)
        self.t_e = numpy.array(
            [[numpy.random.binomial(1, 0.9) for _ in range(mini_batch)]]
        ).astype(numpy.int32).transpose()
        self.gy = numpy.random.uniform(-1, 1, (1, 1))

    # TODO: impl in cpu
    # TODO: divide Gaussian Mixtures and Bernoulli
    def check(self):
        t_x = chainer.Variable(self.context(self.t_x))
        t_e = chainer.Variable(self.context(self.t_e))
        x = chainer.Variable(self.context(self.x))
        loss = gravesnet.loss_func(self.ngauss, x, t_x, t_e)
        loss.backward()
        self.assertEqual(None, t_x.grad)
        self.assertEqual(None, t_e.grad)

        func = loss.creator
        f = lambda: func.forward((self.ngauss, x.data, t_x.data, t_e.data))
        gx, = gradient_check.numerical_grad(f, (x.data,), (loss.grad,))
        gradient_check.assert_allclose(gx, x.grad)

    # def test_gaussian_mixture_2d_cpu(self):
    #     self.context = lambda x: x
    #     self.check()

    @condition.retry(3)
    @attr.gpu
    def test_gaussian_mixture_2d_gpu(self):
        self.context = cuda.to_gpu
        self.check()


testing.run_module(__name__, __file__)
