import unittest
import numpy as np

from othello.helper import judge, extract_valid_hand, is_valid_hand
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


class TestExtractHand(unittest.TestCase):
    def test_true_side(self):
        board = Board.init_board()
        actual = extract_valid_hand(board, True)
        expected = [(2, 4), (4, 2), (3, 5), (5, 3)]
        self.assertEquals(set(expected), set(i.hand for i in actual))

    def test_false_side(self):
        board = Board.init_board()
        actual = extract_valid_hand(board, False)
        expected = [(2, 3), (3, 2), (4, 5), (5, 4)]
        self.assertEquals(set(expected), set(i.hand for i in actual))


class TestIsValidHand(unittest.TestCase):
    def test_valid(self):
        board = Board.init_board()
        actual = is_valid_hand((3, 5), True, board)
        self.assertTrue(actual)

    def test_invalid(self):
        board = Board.init_board()
        actual = is_valid_hand((3, 5), False, board)
        self.assertFalse(actual)

    def test_index_array_of_bound(self):
        board = Board(np.ones((8, 8)))
        board.board[1][1] = 0
        # errorが起きないことを確認
        actual = is_valid_hand((1, 1), False, board)
        self.assertFalse(actual)