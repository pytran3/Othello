from typing import List, Union, Tuple

from othello.model import Board, ScoreBoard, Hand


def judge(board: Board) -> int:
    return board.board.sum()


def simple_score(board: Board, score: ScoreBoard) -> float:
    return (board.board * score.board).sum()


def extract_valid_hand(board: Board, side: bool) -> List[Hand]:
    ret = []
    for i in range(8):
        for j in range(8):
            if is_valid_hand((i, j), side, board):
                ret.append(Hand((i, j), board))
    return ret


def is_valid_hand(hand: Union[Hand, Tuple[int, int]], side: bool, board: Board) -> bool:
    # unwrap
    if isinstance(hand, Hand):
        hand = hand.hand
    board = board.board

    if board[hand[0]][hand[1]] != 0:
        return False
    side_num = 1 if side else -1
    for slide in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        next_point = (hand[0] + slide[0], hand[1] + slide[1])
        while _is_on_board(next_point):
            if board[next_point[0]][next_point[1]] == 0:
                # 囲めない
                next_point = [-1, -1]
                break
            if board[next_point[0]][next_point[1]] != side_num:
                # 裏返されるやつ終わり
                next_point = [i + j for i, j in zip(next_point, slide)]
                break
            next_point = (next_point[0] + slide[0], next_point[1] + slide[1])
        if _is_on_board(next_point) and board[next_point[0]][next_point[1]] == side_num:
            return True
    return False


def _is_on_board(hand: Tuple[int, int]):
    return all([0 <= i < 8 for i in hand])