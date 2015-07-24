from unittest import TestCase


from src.gravesnet import GravesPredictionNet


class TestGravesPredictionNet(TestCase):

    def setUp(self):
        self.model = GravesPredictionNet()

    def test_forward_one_step(self):
        self.fail("not test yet")
