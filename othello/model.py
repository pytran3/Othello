import numpy as np

from othello.exceptions import IllegalShapeException


class Board:
    def __init__(self, board: np.ndarray):
        if board.shape != (8, 8):
            raise IllegalShapeException((8, 8), board.shape)
        self.board = board


class ScoreBoard:
    def __init__(self, board: Board):
        self.board = board.board
