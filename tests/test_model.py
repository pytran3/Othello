import unittest
import numpy as np

from othello.exceptions import IllegalShapeException, IllegalIndexException
from othello.model import Board, Hand


class TestBoard(unittest.TestCase):
    def test_invalid_shape(self):
        with self.assertRaises(IllegalShapeException):
            Board(np.array([[1] * 8, [-1] * 8] * 5))


class TestHand(unittest.TestCase):
    def test_invalid_shape(self):
        with self.assertRaises(IllegalIndexException):
            Hand((0, -1), Board.init_board())
