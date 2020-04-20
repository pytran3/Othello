from abc import abstractmethod, ABC
from typing import List

import numpy as np

from othello.helper import extract_valid_hand
from othello.model import Board, Hand
from othello.search import Searcher, MonteCarloSearcher


class AI(ABC):
    @abstractmethod
    def put(self, board: Board, hands: List[Hand]) -> Hand:
        pass


class RandomAI(AI):
    def put(self, board: Board, hands: List[Hand]) -> Hand:
        hands = extract_valid_hand(board)
        return np.random.choice(hands)


class MiniMaxAI(AI):
    def put(self, board: Board, hands: List[Hand]) -> Hand:
        depth = self.depth if 60 - len(hands) > self.exhaust_threshold else 100
        best_hands, best_score = self.searcher.search_alpha_beta(
            board,
            lambda b: float((b.board * self.score_board).sum()) * (1 if b.side else -1),
            depth,
        )
        return best_hands[0]

    def __init__(self, depth=4, exhaust_threshold=8):
        self.searcher = Searcher()
        self.depth = depth
        self.exhaust_threshold = exhaust_threshold
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
    def put(self, board: Board, hands: List[Hand]) -> Hand:
        if 60 - len(hands) > self.exhaust_threshold:
            best_hand, best_score = self.searcher.search_monte_carlo(
                board,
                self.play_count
            )
        else:
            best_hands, best_score = self.searcher.search_alpha_beta(
                board,
                lambda b: float((b.board * self.score_board).sum()),
                100,
            )
            best_hand = best_hands[0]
        return best_hand

    def __init__(self, exhaust_threshold=8, play_count=30, c=1.0):
        super().__init__()
        self.searcher = MonteCarloSearcher(c)
        self.play_count = play_count
        self.exhaust_threshold = exhaust_threshold
