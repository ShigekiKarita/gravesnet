import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from src.functions.peephole_lstm import peephole_lstm

if cuda.available:
    cuda.init()


def _sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


class TestPeepholeLSTM(unittest.TestCase):

    def setUp(self):
        self.c_prev = numpy.random.uniform(-1,
                                           1, (3, 2, 4)).astype(numpy.float32)
        self.a = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)
        self.x = numpy.random.uniform(-1, 1, (3, 6, 4)).astype(numpy.float32)

        self.gc = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)
        self.gh = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)

    def flat(self):
        self.c_prev = self.c_prev[:, :, 0].copy()
        self.a = self.a[:, :, 0].copy()
        self.x = self.x[:, :, 0].copy()
        self.gc = self.gc[:, :, 0].copy()
        self.gh = self.gh[:, :, 0].copy()

    def check_forward(self, c_prev_data, a_data, x_data):
        c_prev = chainer.Variable(c_prev_data)
        a = chainer.Variable(a_data)
        x = chainer.Variable(x_data)
        c, h = peephole_lstm(c_prev, a, x)
        self.assertEqual(c.data.dtype, numpy.float32)
        self.assertEqual(h.data.dtype, numpy.float32)

        # Compute expected out
        a_in = self.a
        i_in = self.x[:, [0, 3]]
        f_in = self.x[:, [1, 4]]
        o_in = self.x[:, [2, 5]]

        c_expect = _sigmoid(i_in) * numpy.tanh(a_in) + \
            _sigmoid(f_in) * self.c_prev
        h_expect = _sigmoid(o_in) * numpy.tanh(c_expect)

        gradient_check.assert_allclose(c_expect, c.data)
        gradient_check.assert_allclose(h_expect, h.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.c_prev, self.a, self.x)

    @condition.retry(3)
    def test_flat_forward_cpu(self):
        self.flat()
        self.test_forward_cpu()

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.c_prev), cuda.to_gpu(self.a), cuda.to_gpu(self.x))

    @attr.gpu
    @condition.retry(3)
    def test_flat_forward_gpu(self):
        self.flat()
        self.test_forward_gpu()

    def check_backward(self, c_prev_data, a_data, x_data, c_grad, h_grad):
        c_prev = chainer.Variable(c_prev_data)
        a = chainer.Variable(a_data)
        x = chainer.Variable(x_data)
        c, h = peephole_lstm(c_prev, a, x)
        c.grad = c_grad
        h.grad = h_grad
        c.backward()

        func = c.creator
        f = lambda: func.forward((c_prev.data, a.data, x.data))
        gc_prev, ga, gx = gradient_check.numerical_grad(
            f, (c_prev.data, a.data, x.data), (c_grad, h_grad), eps=1e-2)

        gradient_check.assert_allclose(gc_prev, c_prev.grad)
        gradient_check.assert_allclose(ga, a.grad)
        gradient_check.assert_allclose(gx, x.grad)

    @condition.retry(3)
    def test_full_backward_cpu(self):
        self.check_backward(self.c_prev, self.a, self.x, self.gc, self.gh)

    @condition.retry(3)
    def test_flat_full_backward_cpu(self):
        self.flat()
        self.test_full_backward_cpu()

    @condition.retry(3)
    def test_no_gc_backward_cpu(self):
        self.check_backward(self.c_prev, self.a, self.x, None, self.gh)

    @condition.retry(3)
    def test_flat_no_gc_backward_cpu(self):
        self.flat()
        self.test_no_gc_backward_cpu()

    @condition.retry(3)
    def test_no_gh_backward_cpu(self):
        self.check_backward(self.c_prev, self.a, self.x, self.gc, None)

    @condition.retry(3)
    def test_flat_no_gh_backward_cpu(self):
        self.flat()
        self.test_no_gh_backward_cpu()

    @attr.gpu
    @condition.retry(3)
    def test_full_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.c_prev), cuda.to_gpu(self.a), cuda.to_gpu(self.x),
            cuda.to_gpu(self.gc), cuda.to_gpu(self.gh))

    @attr.gpu
    @condition.retry(3)
    def test_flat_full_backward_gpu(self):
        self.flat()
        self.test_full_backward_gpu()

    @attr.gpu
    @condition.retry(3)
    def test_no_gc_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.c_prev), cuda.to_gpu(self.a), cuda.to_gpu(self.x),
            None, cuda.to_gpu(self.gh))

    @attr.gpu
    @condition.retry(3)
    def test_flat_no_gc_backward_gpu(self):
        self.flat()
        self.test_no_gc_backward_gpu()

    @attr.gpu
    @condition.retry(3)
    def test_no_gh_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.c_prev), cuda.to_gpu(self.a), cuda.to_gpu(self.x),
            cuda.to_gpu(self.gc), None)

    @attr.gpu
    @condition.retry(3)
    def test_flat_no_gh_backward_gpu(self):
        self.flat()
        self.test_no_gh_backward_gpu()


testing.run_module(__name__, __file__)
