import unittest

import numpy
from numpy.random import uniform, binomial

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr, condition

from src.gaussian_mixture_2d import gaussian_mixture_2d
from src.gaussian_mixture_2d_ref import gaussian_mixture_2d_ref
from src import gravesnet


if cuda.available:
    cuda.init()


class TestGaussianMixture2d(unittest.TestCase):

    def setUp(self):
        # each 2D-Gaussian contains 6 params: weight, mean(2), stddev(2), corr
        self.ngauss = numpy.random.randint(1, 5)
        input_size = 6 * self.ngauss + 1
        mini_batch = numpy.random.randint(1, 5)
        self.x   = uniform(-1, 1, (mini_batch, input_size)).astype(numpy.float32)
        self.t_x = uniform(-1, 1, (mini_batch, 2)).astype(numpy.float32)
        b_rand   = [[binomial(1, 0.9) for _ in range(mini_batch)]]
        self.t_e = numpy.array(b_rand).astype(numpy.int32).reshape((mini_batch, 1))
        self.g = uniform(-1, 1, (mini_batch, self.ngauss)).astype(numpy.float32)

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
        loss.grad *= 10.0
        # gx, = gradient_check.numerical_grad(f, (x.data,), (loss.grad,), eps=1e-2)
        return x.grad, loss.data

    def check_ref(self):
        t_x = chainer.Variable(self.context(self.t_x))
        t_e = chainer.Variable(self.context(self.t_e))
        x = chainer.Variable(self.context(self.x))
        y, e = gravesnet.split_args(self.ngauss, x, t_x, t_e)
        p = gaussian_mixture_2d(*y)
        q = gaussian_mixture_2d_ref(*y)
        gradient_check.assert_allclose(p.data, q.data)


        # TODO: Check and pass backward
        # x.grad = None
        # p_loss = gravesnet.concat_losses(p, e, t_e)
        # q_loss = gravesnet.concat_losses(q, e, t_e)
        # p_loss.backward()
        # p_xg = x.grad.copy()
        # x.grad = None
        # q_loss.backward()
        # q_xg = x.grad.copy()
        # print(p_xg, q_xg)
        # gradient_check.assert_allclose(p_loss.data, q_loss.data)
        # gradient_check.assert_allclose(p_xg, q_xg)

    @condition.retry(3)
    def test_original_versus_chainer_cpu(self):
        self.context = lambda x: x
        self.check_ref()

    @condition.retry(3)
    @attr.gpu
    def test_original_versus_chainer_gpu(self):
        self.context = cuda.to_gpu
        self.check_ref()

    @condition.retry(3)
    @attr.gpu
    def test_cpu_versus_gpu(self):
        self.context = lambda x: x
        cpu, closs = self.check()
        self.context = cuda.to_gpu
        gpu, gloss = self.check()
        numpy.testing.assert_almost_equal(closs, cuda.to_cpu(gloss))
        gradient_check.assert_allclose(gpu, cpu)


testing.run_module(__name__, __file__)
