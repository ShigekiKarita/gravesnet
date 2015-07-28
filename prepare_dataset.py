import sys
from unittest import TestCase

import numpy
import pickle

from src.dataset import parse_IAMxml, parse_IAMtxt, parse_IAMdataset

class TestParseIAMxml(TestCase):
    def test_parse_IAMxml(self):
        s, e = parse_IAMxml("res/strokesz.xml")
        eps = 1e-5
        numpy.testing.assert_allclose(numpy.mean(s, axis=0), numpy.zeros_like(s[0]), atol=eps)
        numpy.testing.assert_allclose(numpy.std(s, axis=0), numpy.ones_like(s[0]), atol=eps)

    def test_parse_IAMtxt(self):
        d = "/home/karita/Documents/IAM/"
        s, e = parse_IAMdataset(d + "task1/_tes.txt", d + "data/lineStrokes")
        eps = 1e-7
        numpy.testing.assert_allclose(numpy.mean(s, axis=0), numpy.zeros_like(s[0]), atol=eps)
        numpy.testing.assert_allclose(numpy.std(s, axis=0), numpy.ones_like(s[0]), atol=eps)


if __name__ == '__main__':
    d = "/home/karita/Documents/IAM/"
    files = ["testset_v", "testset_t", "testset_f", "trainset"]
    f = sys.argv[1]
    o = parse_IAMdataset(d + "task1/" + f + ".txt", d + "data/lineStrokes")
    pickle.dump(o, open(f + ".npy", "wb"), -1)
