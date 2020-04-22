from abc import abstractmethod, ABC
from typing import List, Tuple

import numpy as np

from othello.helper import extract_valid_hand
from othello.model import Board, Hand, Node
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


class AlphaZero(AI):
    def put(self, board: Board, hands: List[Hand]) -> Hand:
        best_hand, best_score = self.searcher.search_monte_carlo(
            board,
            self.play_count
        )
        return best_hand

    def _predict(self, board: Board):
        board = np.array([[board.board == 1, board.board == -1]])
        p, v = self.network.predict(board)
        return p[0, :], v

    def _calc_valid_hand_p(self, p: np.ndarray, board: Board) -> Tuple[np.ndarray, List[Hand]]:
        p = p[:-1].reshape((8, 8))
        hands = extract_valid_hand(board)
        valid_p = np.array([p[hand.hand[0]][hand.hand[1]] for hand in hands])
        result = valid_p / sum(valid_p + 1e-18)
        return result, hands

    def _evaluate_board(self, board: Board):
        p, v = self._predict(board)
        p, hands = self._calc_valid_hand_p(p, board)
        return p, hands, v

    def __init__(self, network, exhaust_threshold=8, play_count=30):
        super().__init__()
        self.network = network
        self.play_count = play_count
        self.exhaust_threshold = exhaust_threshold

        def select_node(node: Node):
            p, hands, _ = self._evaluate_board(node.board)
            hand = np.random.choice(hands, p=p)
            return [node for node in node.children if node.hand.hand == hand.hand][0]

        def evaluete(board: Board):
            _, _, v = self._evaluate_board(board)
            return v

        self.searcher = MonteCarloSearcher(evaluate=evaluete, select_node=select_node)
