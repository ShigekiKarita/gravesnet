from unittest import TestCase

from chainer import cuda
from chainer.testing import attr
from nose.plugins.attrib import attr as nattr
from chainer import computational_graph as c
import numpy

from src.gravesnet import GravesPredictionNet
from src.dataset import parse_IAMxml

if cuda.available:
    cuda.init()

class TestGravesPredictionNet(TestCase):

    def setUp(self):
        self.model = GravesPredictionNet()
        self.mini_batch = 1
        self.shape = (self.mini_batch, 100)
        self.xs, self.es = parse_IAMxml("res/strokesz.xml")
        self.xs = numpy.float32(self.xs)
        self.es = numpy.float32(self.es)

    def check_one_step(self):
        x = self.context(numpy.concatenate((self.xs[0], self.es[0]))).reshape(1, 3)
        t_x = self.context(self.xs[1]).reshape(1, 2)
        t_e = self.context(numpy.array(self.es[1]).astype(numpy.int32)).reshape(1, 1)
        self.h = self.model.initial_state(self.mini_batch, self.context, "h")
        self.l = self.model.initial_state(self.mini_batch, self.context, "c")
        self.h, self.l, self.loss = self.model.forward_one_step(self.h, self.l, x, t_x, t_e)
        # self.loss.backward()

    @nattr(local=True)
    def test_forward_cpu(self):
        self.mod = numpy
        self.context = lambda x: x
        self.check_one_step()
        with open("gravesnet.dot", "w") as o:
            o.write(c.build_computational_graph((self.loss,), False).dump())

    @nattr(local=True)
    @attr.gpu
    def test_forward_gpu(self):
        self.model.to_gpu()
        self.mod = cuda
        self.context = cuda.to_gpu
        self.check_one_step()
