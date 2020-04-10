from typing import Callable, Tuple, List

from othello.helper import extract_valid_hand, put_and_reverse
from othello.model import Board, Hand


def search_min_max(board: Board, calc_score: Callable[[Board], float], depth: int = 8) -> Tuple[List[Hand], float]:
    if depth == 0:
        return [], calc_score(board)
    best_hands, best_score = None, -1e18
    for point in extract_valid_hand(board):
        new_board = put_and_reverse(point, board)
        hands, score = search_min_max(new_board, calc_score, depth-1)
        if best_score < score:
            best_hands, best_score = hands + [point], score
    return best_hands, best_score
