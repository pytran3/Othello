import unittest

import numpy as np

from othello.model import Board, Node
from othello.parameters import WIN_SCORE
from othello.search import Searcher, MonteCarloSearcher, select_node_ucb


class TestWithReverse(unittest.TestCase):
    def setUp(self) -> None:
        self.searcher = Searcher()

    def _test(self, board: Board, test_func):
        test_func(board)
        board = Board(board.board * -1, not board.side)
        test_func(board)


class TestSearchMinMax(TestWithReverse):
    def test_score(self):
        def test(board: Board):
            def score(b: Board):
                return float(b.board[4][5] * 100)

            actual_hands, actual_score = self.searcher.search_mini_max(board, score, 1)
            actual_hands = [hand.hand for hand in actual_hands]
            self.assertEqual(actual_score, 100)
            self.assertEqual(actual_hands, [(4, 5)])

        board = Board.init_board()
        self._test(board, test)

    def test_depth3(self):
        def test(board):
            def score(b: Board):
                sb = np.zeros((8, 8))
                sb[3][2] = 500
                sb[4][5] = 1000
                sb[5][5] = 50
                sb[6][5] = 10
                sb[3][5] = 45
                sb[2][6] = 1
                return (b.board * sb).sum()

            actual_hands, actual_score = self.searcher.search_mini_max(board, score, 3)
            actual_hands = [hand.hand for hand in actual_hands]
            self.assertEqual(actual_score, 1046)
            self.assertEqual(actual_hands, [(4, 5), (3, 5), (2, 6)])

        board = Board.init_board()
        self._test(board, test)

    def test_depth2(self):
        def test(board):
            def score(b: Board):
                sb = np.zeros((8, 8))
                sb[3][2] = 500
                sb[4][5] = 1000
                sb[5][5] = 50
                sb[6][5] = 10
                sb[3][5] = 45
                sb[2][6] = 1
                return (b.board * sb).sum()

            actual_hands, actual_score = self.searcher.search_mini_max(board, score, 2)
            actual_hands = [hand.hand for hand in actual_hands]
            self.assertEqual(actual_score, 950)
            self.assertEqual(actual_hands, [(4, 5), (5, 5)])

        board = Board.init_board()
        self._test(board, test)

    def test_cant_put(self):
        def test(board):
            def score(_: Board):
                return 0

            actual_hands, actual_score = self.searcher.search_mini_max(board, score, 3)
            actual_hands = [hand.hand for hand in actual_hands]
            self.assertEqual(actual_score, WIN_SCORE)
            self.assertIn(actual_hands, [[(4, 5)], [(5, 4)], [(5, 5)]])

        board = Board.init_board()
        board.board[3][3] = 1
        self._test(board, test)

    def test_select_early_win(self):
        def test(board):
            def score(_: Board):
                return 0

            actual_hands, actual_score = self.searcher.search_mini_max(board, score, 3)
            self.assertEqual(actual_score, WIN_SCORE)
            actual_hands = [hand.hand for hand in actual_hands]
            self.assertEqual(actual_hands, [(2, 2)])

        board = np.zeros((8, 8))
        board[0][2] = 1
        board[1][1] = 1
        board[2][0] = 1
        board[2][1] = -1
        board[1][2] = -1
        board = Board(board)
        self._test(board, test)

    def test_pass(self):
        def test(board):
            def score(_: Board):
                return 0

            actual_hands, actual_score = self.searcher.search_mini_max(board, score, 5)
            self.assertEqual(actual_score, WIN_SCORE)
            self.assertTrue(actual_hands[1].is_pass_hand)
            actual_hands = [hand.hand for hand in actual_hands]
            self.assertIn(actual_hands, [[(0, 2), (0, 0), (7, 2)], [(7, 2), (0, 0), (0, 2)]])

        board = Board(np.zeros((8, 8)))
        board.board[0][0] = 1
        board.board[0][1] = -1
        board.board[7][0] = 1
        board.board[7][1] = -1
        self._test(board, test)


class TestSearchAlphaBeta(TestWithReverse):
    def test_random(self):
        def test(board):
            for i in range(5):
                _sb = np.random.normal(size=(8, 8))

                def score(b: Board):
                    return float((b.board * _sb).sum())

                expected_hands, expected_score = self.searcher.search_mini_max(board, score, 4)
                actual_hands, actual_score = self.searcher.search_alpha_beta(board, score, 4)
                actual_hands = [hand.hand for hand in actual_hands]
                expected_hands = [hand.hand for hand in expected_hands]
                self.assertEqual(expected_score, actual_score)
                self.assertEqual(expected_hands, actual_hands)

        board = Board.init_board()
        self._test(board, test)


class TestSearchMonteCarlo(TestWithReverse):
    def setUp(self) -> None:
        self.searcher = MonteCarloSearcher()

    def test(self):
        def test(board):
            self.searcher.search_monte_carlo(board, 20)

        board = Board.init_board()
        self._test(board, test)

    def test_select_node(self):
        nodes = [Node(Board.init_board()) for _ in range(4)]
        for i in range(3):
            nodes[i].w = 2
            nodes[i].n = 4
        actual = select_node_ucb(nodes, 1.0)
        expected = nodes[-1]
        self.assertEqual(actual, expected)

    def test_select_node_exploit(self):
        nodes = [Node(Board.init_board()) for _ in range(4)]
        for i in range(3):
            nodes[i].w = 2
            nodes[i].n = 4
        nodes[-1].w = 1000
        nodes[-1].n = 1000
        actual = select_node_ucb(nodes, 0.0)
        expected = nodes[-1]
        self.assertEqual(actual, expected)

    def test_select_node_explore(self):
        nodes = [Node(Board.init_board()) for _ in range(4)]
        for i in range(3):
            nodes[i].w = 2
            nodes[i].n = 4
        nodes[-1].w = 3
        nodes[-1].n = 1000
        actual = select_node_ucb(nodes, 1.0)
        expected = nodes[-1]
        self.assertNotEqual(actual, expected)
