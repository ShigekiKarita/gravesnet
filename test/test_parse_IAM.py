from unittest import TestCase

import numpy

from src.dataset import parse_IAMxml, extract_raw_text, extract_ascii


class TestParseIAMxml(TestCase):
    def setUp(self):
        self.file = "res/strokesz.xml"
        self.ascii = "res/a01-000u.txt"

    def test_parse_line_strokes(self):
        s, e = parse_IAMxml(self.file)
        eps = 1e-5
        numpy.testing.assert_allclose(numpy.mean(s, axis=0), numpy.zeros_like(s[0]), atol=eps)
        numpy.testing.assert_allclose(numpy.std(s, axis=0), numpy.ones_like(s[0]), atol=eps)

    def test_parse_line_text(self):
        t = extract_raw_text(self.file)
        self.assertEqual(t[-1], "by Mr. Will Griffiths, MP for")

    def test_extract_ascii(self):
        t = extract_ascii(self.ascii)
        self.assertEqual(t[-1], 'resolution on the subject')