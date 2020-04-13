from typing import Callable, Tuple, List, Union

from othello.helper import extract_valid_hand, put_and_reverse, judge
from othello.model import Board, Hand
from othello.parameters import WIN_SCORE


class Searcher:
    def search_mini_max(self, board: Board, calc_score: Callable[[Board], float], depth: int = 8, pass_flag=False) -> Tuple[List[Hand], float]:
        if depth == 0:
            return [], calc_score(board)
        best_hands, best_score = None, (-1e18 if board.side else 1e18)
        for point in self._extract_valid_hand(board):
            new_board = self._put_and_reverse(point, board)
            hands, score = self.search_mini_max(new_board, calc_score, depth - 1)
            if best_score < score and board.side:
                best_hands, best_score = ([point] + hands), score
            if score < best_score and not board.side:
                best_hands, best_score = ([point] + hands), score
        if best_hands is None:
            if pass_flag:
                return [], (WIN_SCORE if judge(board) >= 0 else -WIN_SCORE)
            board = Board(board.board, not board.side)
            return self.search_mini_max(board, calc_score, depth, True)
        return best_hands, best_score

    def _extract_valid_hand(self, board: Board):
        return extract_valid_hand(board)

    def _put_and_reverse(self, hand: Union[Hand, Tuple[int, int]], board: Board) -> Board:
        return put_and_reverse(hand, board)