import unittest

import numpy as np

from othello.model import Board
from othello.search import Searcher


class TestSearchMinMax(unittest.TestCase):
    def setUp(self) -> None:
        self.searcher = Searcher()

    def test_score(self):
        def score(b: Board):
            return float(b.board[4][5] * 100)

        board = Board.init_board()
        actual_hands, actual_score = self.searcher.search_mini_max(board, score, 1)
        actual_hands = [hand.hand for hand in actual_hands]
        self.assertEqual(actual_score, 100)
        self.assertEqual(actual_hands, [(4, 5)])

    def test_depth3(self):
        def score(b: Board):
            sb = np.zeros((8, 8))
            sb[3][2] = 500
            sb[4][5] = 1000
            sb[5][5] = 50
            sb[6][5] = 10
            sb[3][5] = 100
            return (b.board * sb).sum() * (1 if board.side else -1)

        board = Board.init_board()
        actual_hands, actual_score = self.searcher.search_mini_max(board, score, 3)
        actual_hands = [hand.hand for hand in actual_hands]
        self.assertEqual(actual_score, 1100)
        self.assertEqual(actual_hands, [(4, 5), (3, 5), (2, 5)])
