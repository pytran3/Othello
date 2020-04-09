import unittest
import numpy as np

from othello.helper import judge
from othello.model import Board


class TestJudge(unittest.TestCase):
    def test_even(self):
        even = Board(np.array([[1] * 8, [-1] * 8] * 4))
        even_result = judge(even)
        self.assertEquals(0, even_result)

    def test_win(self):
        win = Board(np.array([[1] * 8, [-1] * 8] * 4))
        win.board[1, 0] = 1
        win_result = judge(win)
        self.assertEquals(2, win_result)