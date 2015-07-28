from unittest import TestCase

import numpy

from src.dataset import parse_IAMxml, parse_IAMtxt

class TestParseIAMxml(TestCase):
    def test_parse_IAMxml(self):
        s, e = parse_IAMxml("res/strokesz.xml")
        eps = 1e-5
        numpy.testing.assert_allclose(numpy.mean(s, axis=0), numpy.zeros_like(s[0]), atol=eps)
        numpy.testing.assert_allclose(numpy.std(s, axis=0), numpy.ones_like(s[0]), atol=eps)

    def test_parse_IAMtxt(self):
        d = "/home/karita/Documents/IAM/"
        s, e = parse_IAMtxt(d + "task1/testset_v.txt",
                            d + "data/lineStrokes")