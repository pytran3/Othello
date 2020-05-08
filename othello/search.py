import math
from typing import Callable, Tuple, List, Union

import numpy as np

from othello.helper import extract_valid_hand, put_and_reverse, is_finished, judge_simple
from othello.model import Board, Hand, Node
from othello.parameters import WIN_SCORE


class Searcher:
    def search_mini_max(self, board: Board, calc_score: Callable[[Board], float], depth: int = 8) -> \
            Tuple[List[Hand], float]:
        if is_finished(board):
            return [], WIN_SCORE * judge_simple(board) * (board.side * 2 - 1)
        if depth == 0:
            return [], calc_score(board) * (board.side * 2 - 1)
        best_hands, best_score = None, -float("inf")
        for point in self._extract_valid_hand(board):
            if point.is_pass_hand:
                new_board = Board(board.board, not board.side)
            else:
                new_board = self._put_and_reverse(point, board)
            hands, score = self.search_mini_max(new_board, calc_score, depth - 1)
            score = -score
            if best_score < score:
                best_hands, best_score = ([point] + hands), score
        return best_hands, best_score

    def search_alpha_beta(
            self,
            board: Board,
            calc_score: Callable[[Board], float],
            depth: int = 8,
            alpha=-float("inf"),
            beta=float("inf")
    ) -> Tuple[List[Hand], float]:
        if is_finished(board):
            return [], WIN_SCORE * judge_simple(board) * (board.side * 2 - 1)
        if depth == 0:
            return [], calc_score(board) * (board.side * 2 - 1)
        best_hands = []
        for point in self._extract_valid_hand(board):
            if point.is_pass_hand:
                new_board = Board(board.board, not board.side)
            else:
                new_board = self._put_and_reverse(point, board)
            hands, score = self.search_alpha_beta(new_board, calc_score, depth - 1, -beta, -alpha)
            score = -score
            if alpha < score:
                best_hands, alpha = ([point] + hands), score
            if beta <= alpha:
                return best_hands, alpha
        return best_hands, alpha

    def _extract_valid_hand(self, board: Board):
        ret = extract_valid_hand(board)
        if ret:
            return ret
        else:
            return [Hand.pass_hand()]

    def _put_and_reverse(self, hand: Union[Hand, Tuple[int, int]], board: Board) -> Board:
        return put_and_reverse(hand, board)


def play_out(board: Board) -> float:
    while not is_finished(board):
        valid_hands = extract_valid_hand(board)
        if len(valid_hands) == 0:
            board = Board(board.board, not board.side)
            continue
        hand = np.random.choice(valid_hands)
        board = put_and_reverse(hand, board)
    return judge_simple(board)


def select_best_node_ucb(node: Node) -> Node:
    return max(node.children, key=lambda x: x.n)


def select_node_ucb(node: Node, c: float = 1.0) -> Node:
    eval_list = eval_nodes_ucb(node.children, c)
    return max(eval_list, key=lambda x: x[0])[1]


def eval_nodes_ucb(nodes: List[Node], c: float):
    def ucb(node: Node):
        return node.w + c * math.sqrt(2 * math.log(n) / (node.n + 1e-8))

    n = sum([node.n for node in nodes])
    return [(ucb(x) if x.n else 1e18, x) for x in nodes]


class MonteCarloSearcher(Searcher):
    def __init__(self, expansion_threshold=3, evaluate=play_out, select_node=select_node_ucb,
                 select_best_node=select_best_node_ucb):
        self.expansion_threshold = expansion_threshold
        self.evaluate = evaluate
        self.select_node = select_node
        self.select_best_node = select_best_node
        self.memo_root = None

    def search_monte_carlo(self, board: Board, play_count=100, hands: List[Hand] = None) -> Tuple[Hand, float]:
        if self.memo_root is None or not hands:
            root_node = Node(board)
        else:
            if [i for i in hands if i.is_pass_hand] and False:
                root_node = Node(board)
            else:
                tmp = [child for child in self.memo_root.children if child.hand.hand == hands[-1].hand and child.hand.is_pass_hand == hands[-1].is_pass_hand]
                if tmp:
                    root_node = tmp[0]
                else:
                    root_node = Node(board)
        # root_node = Node(board)
        # print(root_node.n)
        if len(root_node.children) == 0:
            root_node.children = self._expand(root_node)

        for play_index in range(play_count):
            node = root_node
            while node.children:
                node = self.select_node(node)

            if node.board.is_finished is None:
                node.board.is_finished = is_finished(node.board)
            if node.board.is_finished:
                value = judge_simple(node.board)
            else:
                if node.n >= self.expansion_threshold:
                    # expansion
                    node.children = self._expand(node)
                    node = self.select_node(node)
                value = self.evaluate(node.board)

            node.w += value
            node.n += 1
            while node.parent:
                value = - value
                node = node.parent
                node.w += value
                node.n += 1
        best_node = self.select_best_node(root_node)
        self.memo_root = best_node
        return best_node.hand, best_node.w

    def _expand(self, node: Node) -> List[Node]:
        children = [
            Node(self._put_and_reverse(hand, node.board), node, hand=hand) if not hand.is_pass_hand else Node(
                Board(node.board.board, not node.board.side), node, hand=hand)
            for hand in self._extract_valid_hand(node.board)
        ]
        node.children = children
        return children
