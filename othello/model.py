from typing import Tuple

import numpy as np

from othello.exceptions import IllegalShapeException, IllegalIndexException


class Board:
    def __init__(self, board: np.ndarray, side: bool = True):
        if board.shape != (8, 8):
            raise IllegalShapeException((8, 8), board.shape)
        self.board = board.copy()
        self.side = side

    def __str__(self):
        return "Board (side={}, {})".format(self.side, self.board)

    @staticmethod
    def init_board(side: bool = True):
        board = np.zeros((8, 8), dtype=int)
        board[3][3] = -1
        board[4][4] = -1
        board[3][4] = 1
        board[4][3] = 1
        return Board(board, side)


class ScoreBoard:
    def __init__(self, board: Board):
        self.board = board.board


class Hand:
    def __init__(self, hand: Tuple[int, int], board: Board):
        if (not 0 <= hand[0] < 8) or (not 0 <= hand[1] < 8):
            raise IllegalIndexException((8, 8), hand)
        self.hand = hand
        self.board = board

    def __str__(self):
        return "Hand ({})".format(self.hand)


class Node:
    """
    Node for tree search

    """
    def __init__(self, board: Board, parent=None, hand: Hand = None):
        self.parent = parent
        self.hand = hand
        self.win_count = 0
        self.lose_count = 0
        self.even_count = 0
        self.board = board
        self.children = []

    def win_rate(self):
        return self.win_count / self.count() if self.count() else 0

    def count(self):
        return self.win_count + self.lose_count

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)