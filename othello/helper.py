from typing import List, Union, Tuple

import numpy as np

from othello.exceptions import OthelloRuntimeException
from othello.model import Board, ScoreBoard, Hand

_SLIDES = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def judge(board: Board) -> int:
    return board.board.sum()


def judge_simple(board: Board) -> int:
    ret = judge(board)
    if ret > 0:
        return 1
    elif ret < 0:
        return -1
    else:
        return 0


def simple_score(board: Board, score: ScoreBoard) -> float:
    return (board.board * score.board).sum()


def put_and_reverse(hand: Union[Hand, Tuple[int, int]], board: Board) -> Board:
    if not is_valid_hand(hand, board):
        raise OthelloRuntimeException("invalid hand: {} {}".format(hand, board))
    new_board = Board(board.board.copy(), not board.side)
    hand, side_num, board = _unwrap(hand, board)
    board = new_board.board
    for slide in _SLIDES:
        next_point = (hand[0] + slide[0], hand[1] + slide[1])
        while _is_on_board(next_point):
            if board[next_point[0]][next_point[1]] == 0:
                # 囲めない
                next_point = (-1, -1)
                break
            if board[next_point[0]][next_point[1]] == side_num:
                # 裏返されるやつ終わり
                break
            next_point = (next_point[0] + slide[0], next_point[1] + slide[1])
        if _is_on_board(next_point):
            while next_point != hand:
                next_point = (next_point[0] - slide[0], next_point[1] - slide[1])
                board[next_point[0]][next_point[1]] = side_num
    board[hand[0]][hand[1]] = side_num
    return new_board


def extract_valid_hand(board: Board) -> List[Hand]:
    ret = []
    for i in range(8):
        for j in range(8):
            if is_valid_hand((i, j), board):
                ret.append(Hand((i, j), board))
    return ret


def is_valid_hand(hand: Union[Hand, Tuple[int, int]], board: Board) -> bool:
    hand, side_num, board = _unwrap(hand, board)
    hand_y = hand[0]
    hand_x = hand[1]

    if board[hand_y][hand_x] != 0:
        return False
    for slide_y, slide_x in _SLIDES:
        next_point = (hand_y + slide_y, hand_x + slide_x)
        while _is_on_board(next_point):
            if board[next_point[0]][next_point[1]] == 0:
                # 囲めない
                next_point = [-1, -1]
                break
            if board[next_point[0]][next_point[1]] == side_num:
                # 裏返されるやつ終わり
                break
            next_point = (next_point[0] + slide_y, next_point[1] + slide_x)
        if _is_on_board(next_point) and distance(hand, next_point) > 1:
            return True
    return False


def distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    # return max(abs(a[0] - b[0]), abs(a[1] - b[1]))  # 下の方が速い
    return max(a[0] - b[0], b[0] - a[0], a[1] - b[1], b[1] - a[1])


def is_finished(board: Board):
    if extract_valid_hand(board):
        return False
    board = Board(board.board, not board.side)
    if extract_valid_hand(board):
        return False
    return True


def boltzmann(p: np.ndarray, temperature: float = 0):
    if temperature == 0:
        action = np.argmax(p)
        scores = np.zeros(len(p))
        scores[action] = 1
    else:
        p = p ** (1 / temperature)
        scores = p / p.sum()
    return scores


def _is_on_board(hand: Tuple[int, int]):
    return 0 <= hand[0] < 8 and 0 <= hand[1] < 8


def _unwrap(hand: Union[Hand, Tuple[int, int]], board: Board) -> Tuple[Tuple[int, int], int, np.ndarray]:
    # unwrap
    if isinstance(hand, Hand):
        hand = hand.hand
    side_num = 1 if board.side else -1
    board = board.board
    return hand, side_num, board
