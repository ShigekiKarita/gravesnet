from unittest import TestCase

import numpy

from src.dataset import parse_IAMxml

class TestParseIAMxml(TestCase):
    def test_parse_IAMxml(self):
        s, e = parse_IAMxml("res/strokesz.xml")
        numpy.testing.assert_almost_equal(numpy.mean(s), 0.0, decimal=5)
        numpy.testing.assert_almost_equal(numpy.std(s), 1.0, decimal=5)
