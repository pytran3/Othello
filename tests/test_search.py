import unittest

import numpy as np

from othello.model import Board
from othello.parameters import WIN_SCORE
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

    def test_cant_put(self):
        def score(_: Board):
            return 0

        board = Board.init_board()
        board.board[3][3] = 1
        actual_hands, actual_score = self.searcher.search_mini_max(board, score, 3)
        actual_hands = [hand.hand for hand in actual_hands]
        self.assertEqual(actual_score, WIN_SCORE)
        self.assertIn(actual_hands, [[(4, 5)], [(5, 4)], [(5, 5)]])

    def test_pass(self):
        def score(_: Board):
            return 0

        board = Board(np.zeros((8, 8)))
        board.board[0][0] = 1
        board.board[0][1] = -1
        board.board[7][0] = 1
        board.board[7][1] = -1
        actual_hands, actual_score = self.searcher.search_mini_max(board, score, 3)
        actual_hands = [hand.hand for hand in actual_hands]
        self.assertEqual(actual_score, WIN_SCORE)
        self.assertIn(actual_hands, [[(0, 2), (7, 2)], [(7, 2), (0, 2)]])


class TestSearchAlphaBeta(unittest.TestCase):
    def setUp(self) -> None:
        self.searcher = Searcher()

    def test_random(self):
        for i in range(5):
            _sb = np.random.normal(size=(8, 8))

            def score(b: Board):
                return float((b.board * _sb).sum())

            board = Board.init_board()
            actual_hands, actual_score = self.searcher.search_alpha_beta(board, score, 4)
            expected_hands, expected_score = self.searcher.search_mini_max(board, score, 4)
            actual_hands = [hand.hand for hand in actual_hands]
            expected_hands = [hand.hand for hand in expected_hands]
            self.assertEqual(expected_score, actual_score)
            self.assertEqual(expected_hands, actual_hands)
