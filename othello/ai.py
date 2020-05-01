import math
from abc import abstractmethod, ABC
from typing import List, Tuple

import numpy as np

from othello.helper import extract_valid_hand, boltzmann
from othello.model import Board, Hand, Node
from othello.network import DN_OUTPUT_SIZE
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
                self.play_count,
                hands,
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
            self.play_count,
            # hands
        )
        return best_hand

    def _predict(self, board: Board):
        if board.side is False:
            board = board.board * -1
        else:
            board = board.board
        board = np.array([[board == 1, board == -1]])
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
        if len(p) == 0:
            p = np.ones(1)
            hands = [Hand.pass_hand()]
        return p, hands, v

    def __init__(self, network, c=1.0, temperature=1.0, exhaust_threshold=8, play_count=30, history=False):
        super().__init__()
        self.network = network
        self.c = c
        self.temperature = temperature
        self.play_count = play_count
        self.exhaust_threshold = exhaust_threshold
        self.history = [] if history else None

        def select_node(node: Node):
            if node.children[0].p is None:
                p, hands, _ = self._evaluate_board(node.board)
                p = boltzmann(p)
                tmp = dict(zip([hand.hand for hand in hands], p))
                for child in node.children:
                    child.p = tmp[child.hand.hand]
            t = sum([child.n for child in node.children])
            t = math.sqrt(t)
            arc = [(-child.w / child.n if child.n != 0 else 0.0) + self.c * child.p * t / (child.n + 1) for child in
                   node.children]
            action = np.argmax(arc)
            return node.children[action]

        def evaluate(board: Board):
            _, _, v = self._evaluate_board(board)
            return v

        def select_best_node(node: Node):
            p = np.array([child.n + 1 for child in node.children])
            p = p / p.sum()
            p = boltzmann(p, self.temperature)
            if self.history is not None:
                policies = [0] * DN_OUTPUT_SIZE
                for i in range(len(p)):
                    index = node.children[i].hand.hand[0] * 8 + node.children[i].hand.hand[1]
                    policies[index] = p[i]
                self.history.append([node.board.board, policies, None])
            return np.random.choice([child for child in node.children], p=p)

        self.searcher = MonteCarloSearcher(expansion_threshold=1, evaluate=evaluate, select_node=select_node,
                                           select_best_node=select_best_node)
