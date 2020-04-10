from typing import List, Union, Tuple

import numpy as np

from othello.exceptions import OthelloRuntimeException
from othello.model import Board, ScoreBoard, Hand


def judge(board: Board) -> int:
    return board.board.sum()


def simple_score(board: Board, score: ScoreBoard) -> float:
    return (board.board * score.board).sum()


def put_and_reverse(hand: Union[Hand, Tuple[int, int]], board: Board) -> Board:
    if not is_valid_hand(hand, board):
        raise OthelloRuntimeException("invalid hand: {} {}".format(hand, board))
    new_board = Board(board.board, not board.side)
    hand, side_num, board = _unwrap(hand, board)
    board = new_board.board
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

    if board[hand[0]][hand[1]] != 0:
        return False
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


def _unwrap(hand: Union[Hand, Tuple[int, int]], board: Board) -> Tuple[Tuple[int, int], int, np.ndarray]:
    # unwrap
    if isinstance(hand, Hand):
        hand = hand.hand
    side_num = 1 if board.side else -1
    board = board.board
    return hand, side_num, board
