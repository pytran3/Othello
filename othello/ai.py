from abc import abstractmethod, ABC

import numpy as np

from othello.model import Board, Hand
from othello.search import Searcher, MonteCarloSearcher


class AI(ABC):
    @abstractmethod
    def put(self, board: Board) -> Hand:
        pass


class MiniMaxAI(AI):
    def put(self, board: Board) -> Hand:
        best_hands, best_score = self.searcher.search_alpha_beta(
            board,
            lambda b: float((b.board * self.score_board).sum()),
            depth=self.depth,
        )
        print("Evaluation: {}".format(best_score))
        return best_hands[0]

    def __init__(self, depth=4):
        self.searcher = Searcher()
        self.depth = depth
        self.score_board = np.array(  # https://uguisu.skr.jp/othello/5-1.html
            [
                [30, -12, 0, -1, -1, 0, -12, 30],
                [-12, -15, -3, -3, -3, -3, -15, -12],
                [0, -3, 0, -1, -1, 0, -3, 0],
                [-1, -3, -1, -1, -1, -1, -3, -1],
                [-1, -3, -1, -1, -1, -1, -3, -1],
                [0, -3, 0, -1, -1, 0, -3, 0],
                [-12, -15, -3, -3, -3, -3, -15, -12],
                [30, -12, 0, -1, -1, 0, -12, 30],
            ]
        )


class MonteCarloAI(MiniMaxAI):
    def put(self, board: Board) -> Hand:
        best_hand, best_score = self.searcher.search_monte_carlo(
            board,
            self.play_count
        )
        print("Evaluation: {}".format(best_score))
        return best_hand

    def put_exhaustive_search(self, board: Board) -> Hand:
        return super().put(board)

    def __init__(self, play_count=30, c=1.0):
        super().__init__(100)
        self.searcher = MonteCarloSearcher(c)
        self.play_count = play_count
