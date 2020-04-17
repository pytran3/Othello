import math
from typing import Callable, Tuple, List, Union

import numpy as np

from othello.helper import extract_valid_hand, put_and_reverse, judge, is_finished, judge_simple
from othello.model import Board, Hand, Node
from othello.parameters import WIN_SCORE


class Searcher:
    def search_mini_max(self, board: Board, calc_score: Callable[[Board], float], depth: int = 8) -> \
            Tuple[List[Hand], float]:
        if is_finished(board):
            return [], WIN_SCORE * judge_simple(board)
        if depth == 0:
            return [], -calc_score(board)
        best_hands, best_score = None, -1e18
        for point in self._extract_valid_hand(board):
            if point.is_pass_hand:
                new_board = Board(board.board, not board.side)
            else:
                new_board = self._put_and_reverse(point, board)
            hands, score = self.search_mini_max(new_board, calc_score, depth - 1)
            if best_score < score:
                best_hands, best_score = ([point] + hands), score
        return best_hands, -best_score

    def search_alpha_beta(
            self,
            board: Board,
            calc_score: Callable[[Board], float],
            depth: int = 8,
            neighbor_best=None,
            pass_flag=False
    ) -> Tuple[List[Hand], float]:
        if depth == 0:
            return [], calc_score(board)
        best_hands, best_score = None, (-1e18 if board.side else 1e18)
        for point in self._extract_valid_hand(board):
            new_board = self._put_and_reverse(point, board)
            hands, score = self.search_alpha_beta(new_board, calc_score, depth - 1, neighbor_best)
            if (best_score < score and board.side) or (score < best_score and not board.side):
                best_hands, best_score = ([point] + hands), score
                neighbor_best = score
            if neighbor_best is not None:
                if (neighbor_best < best_score and board.side) or (best_score < neighbor_best and not board.side):
                    return ([point] + hands), score
        if best_hands is None:
            if pass_flag:
                return [], (WIN_SCORE if judge(board) >= 0 else -WIN_SCORE)
            board = Board(board.board, not board.side)
            return self.search_alpha_beta(board, calc_score, depth, True)
        return best_hands, best_score

    def _extract_valid_hand(self, board: Board):
        ret = extract_valid_hand(board)
        if ret:
            return ret
        else:
            return [Hand.pass_hand()]

    def _put_and_reverse(self, hand: Union[Hand, Tuple[int, int]], board: Board) -> Board:
        return put_and_reverse(hand, board)


class MonteCarloSearcher(Searcher):
    def __init__(self, c=1.0):
        self.c = c

    def search_monte_carlo(self, board: Board, play_count=100) -> Tuple[Hand, float]:
        root_node = Node(board)
        root_children = self._expand(root_node)
        [self._expand(child) for child in root_children]
        leaf_nodes = root_children.copy()

        for play_index in range(play_count):
            node = self._select_node(leaf_nodes, play_index + 1)
            if len(node.children) == 0:
                node.value = judge_simple(node.board)
            elif node.n > 2:
                # expansion
                new_leaf_nodes = self._expand(node)
                del leaf_nodes[leaf_nodes.index(node)]
                node = self._select_node(new_leaf_nodes, play_index + 1)
                leaf_nodes.extend(new_leaf_nodes)
                node.w += self._play_out(node.board)
            else:
                node.w += self._play_out(node.board)
            self._backward(root_node)
        best_node = max(root_children, key=lambda x: x.n)
        return best_node.hand, best_node.w

    def _expand(self, node: Node) -> List[Node]:
        ret = [
            Node(self._put_and_reverse(hand, node.board), node, hand=hand)
            for hand in self._extract_valid_hand(node.board)
        ]
        node.children = ret
        return ret

    def _play_out(self, board: Board) -> int:
        while not is_finished(board):
            valid_hands = self._extract_valid_hand(board)
            if valid_hands[0].is_pass_hand:
                board = Board(board.board, not board.side)
                continue
            hand = np.random.choice(valid_hands)
            board = put_and_reverse(hand, board)
        return judge_simple(board)

    def _select_node(self, leaf_nodes: List[Node], n: int) -> Node:
        eval_list = self._eval_nodes(leaf_nodes, n)
        return max(eval_list, key=lambda x: x[0])[1]

    def _eval_nodes(self, nodes: List[Node], n: int):
        def ucb(node: Node):
            return node.w + self.c * math.sqrt(2 * math.log(n) / (node.n + 1e-8))

        return [(ucb(x), x) for x in nodes]

    def _backward(self, root: Node) -> float:
        if len(root.children):
            value = root.value
            root.value = 0
            root.w += value
            root.n += 1
            return value
        n = sum([x.n for x in root.children])
        best_node = max([x for x in self._eval_nodes(root.children, n)], key=lambda x: x[0])[1]
        value = -self._backward(best_node)
        root.w += value
        root.n += 1
        return value
