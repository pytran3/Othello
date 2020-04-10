import unittest

from othello.model import Board
from othello.search import search_min_max


class TestSearchMinMax(unittest.TestCase):
    def test_score(self):
        def score(board: Board):
            return float(board.board[3][5] * 100)
        board = Board.init_board()
        actual_hands, actual_score = search_min_max(board, score, 1)
        actual_hands = [hand.hand for hand in actual_hands]
        self.assertEquals(actual_score, 100)
        self.assertEquals(actual_hands, [(3, 5)])
