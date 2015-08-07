from unittest import TestCase

import chainer
from chainer import testing
from chainer.gradient_check import assert_allclose
from chainer import cuda
from chainer.testing import attr, condition
import numpy

from src.functions.vec_sub_mat import vec_sub_mat

if cuda.available:
    cuda.init()


class TestVecSubMat(TestCase):

    def setUp(self):
        r = numpy.random.randint(2, 5)
        c = numpy.random.randint(2, 5)
        self.v = numpy.arange(r).reshape(r, 1).astype(numpy.float32)
        self.n = numpy.random.uniform(-1, 1, (r, c)).astype(numpy.float32)
        self.m = numpy.full_like(self.n, self.v) - self.n
        self.g = numpy.arange(r * c).reshape(r, c).astype(numpy.float32)
        self.h = self.g.sum(1).reshape(r, 1)

    def check(self, context):
        v = chainer.Variable(context(self.v))
        m = chainer.Variable(context(self.m))
        n = vec_sub_mat(v, m)
        assert_allclose(n.data, context(self.n))

        n.grad = context(self.g)
        n.backward()
        assert_allclose(m.grad, -n.grad)
        assert_allclose(v.grad, context(self.h))

    @condition.retry(100)
    def test_cpu(self):
        self.check(lambda x: x)

    @condition.retry(100)
    @attr.gpu
    def test_gpu(self):
        self.check(cuda.to_gpu)

testing.run_module(__name__, __file__)