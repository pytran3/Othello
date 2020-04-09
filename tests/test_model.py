import unittest
import numpy as np

from othello.exceptions import IllegalShapeException
from othello.model import Board


class TestBoard(unittest.TestCase):
    def test_invalid_shape(self):
        with self.assertRaises(IllegalShapeException):
            Board(np.array([[1] * 8, [-1] * 8] * 5))