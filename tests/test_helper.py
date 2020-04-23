import unittest
import numpy as np

from othello.helper import judge, extract_valid_hand, is_valid_hand, put_and_reverse, is_finished, boltzmann
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
        actual = extract_valid_hand(board)
        expected = [(2, 3), (3, 2), (4, 5), (5, 4)]
        self.assertEquals(set(expected), set(i.hand for i in actual))

    def test_false_side(self):
        board = Board.init_board(False)
        actual = extract_valid_hand(board)
        expected = [(2, 4), (4, 2), (3, 5), (5, 3)]
        self.assertEquals(set(expected), set(i.hand for i in actual))


class TestIsValidHand(unittest.TestCase):
    def test_valid(self):
        board = Board.init_board()
        actual = is_valid_hand((4, 5), board)
        self.assertTrue(actual)

    def test_invalid(self):
        board = Board.init_board(False)
        actual = is_valid_hand((4, 5), board)
        self.assertFalse(actual)

    def test_index_array_of_bound(self):
        board = Board(np.ones((8, 8)), False)
        board.board[1][1] = 0
        # errorが起きないことを確認
        actual = is_valid_hand((1, 1), board)
        self.assertFalse(actual)

    def test_diagonal(self):
        board = Board.init_board()
        board.board[3][3] = 1
        actual = is_valid_hand((5, 5), board)
        self.assertTrue(actual)

    def test_jump(self):
        board = Board.init_board()
        board.board[4][5] = 1
        actual = is_valid_hand((4, 6), board)
        self.assertFalse(actual)


class TestIsFinished(unittest.TestCase):
    def test_finish(self):
        board = Board(np.zeros((8, 8)))
        actual = is_finished(board)
        self.assertTrue(actual)

    def test_able_o(self):
        board = Board(np.zeros((8, 8)))
        board.board[0][0] = 1
        board.board[0][1] = -1
        actual = is_finished(board)
        self.assertFalse(actual)

    def test_able_x(self):
        board = Board(np.zeros((8, 8)), False)
        board.board[0][0] = 1
        board.board[0][1] = -1
        actual = is_finished(board)
        self.assertFalse(actual)


class TestPutAndReverse(unittest.TestCase):
    def test_reverse(self):
        board = Board.init_board()
        actual = put_and_reverse((4, 5), board)
        expected = Board.init_board().board
        expected[4][4] = 1
        expected[4][5] = 1
        np.testing.assert_array_equal(expected, actual.board)

    def test_index_array_of_bound(self):
        board = Board(np.zeros((8, 8)))
        board.board[0][0] = -1
        board.board[0][2] = -1
        board.board[0][3] = 1
        actual = put_and_reverse((0, 1), board)
        expected = np.zeros_like(board.board)
        expected[0][0] = -1
        expected[0][1] = 1
        expected[0][2] = 1
        expected[0][3] = 1
        np.testing.assert_array_equal(expected, actual.board)

    def test_multi_size_reverse(self):
        board = Board.init_board()
        board.board[4][5] = -1
        actual = put_and_reverse((4, 6), board)
        expected = Board.init_board().board
        expected[4][4] = 1
        expected[4][5] = 1
        expected[4][6] = 1
        np.testing.assert_array_equal(expected, actual.board)


class TestBoltzmann(unittest.TestCase):
    def test_0_temperature(self):
        p = np.array([1, 2, 3])
        actual = boltzmann(p, 0)
        expected = np.array([0, 0, 1])
        np.testing.assert_equal(actual, expected)

    def test_1_temperature(self):
        p = np.array([1, 2, 3])
        actual = boltzmann(p, 1)
        expected = np.array([1/6, 2/6, 3/6])
        np.testing.assert_equal(actual, expected)