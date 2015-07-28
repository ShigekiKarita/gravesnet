from unittest import TestCase

import numpy
from nose.plugins.attrib import attr

from src.dataset import parse_IAMxml, parse_IAMtxt

class TestParseIAMxml(TestCase):
    @attr(local=True)
    def test_parse_IAMxml(self):
        s, e = parse_IAMxml("res/strokesz.xml")
        eps = 1e-5
        numpy.testing.assert_allclose(numpy.mean(s, axis=0), numpy.zeros_like(s[0]), atol=eps)
        numpy.testing.assert_allclose(numpy.std(s, axis=0), numpy.ones_like(s[0]), atol=eps)
