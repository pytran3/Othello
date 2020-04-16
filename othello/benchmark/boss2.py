from typing import Tuple

from othello.ai import MonteCarloAI
from othello.helper import extract_valid_hand, is_valid_hand, put_and_reverse
from othello.model import Board, Hand
from othello.view import view_board


def main(sente=True):
    board = Board.init_board()
    board.side = sente
    ai = MonteCarloAI()
    print(view_board(board))
    while True:
        if not extract_valid_hand(board):
            board.side ^= True
        if not extract_valid_hand(board):
            break
        if board.side:
            hand = input_hand(board)
        else:
            if (board.board == 0).sum() < 8:
                # 計算時間に余裕があるので全探索
                hand = ai.put_exhaustive_search(board)
            else:
                hand = ai.put(board)
            print("AI put: {}".format(hand))

        board = put_and_reverse(hand, board)
        print(view_board(board))

    print("=" * 10 + "  GAME OVER  " + "=" * 10)
    x_count = (board.board == 1).sum()
    o_count = (board.board == -1).sum()
    print("x: {}, o: {}".format(x_count, o_count))


def input_hand(board: Board):
    while True:
        try:
            hand = parse(input())
        except Exception:
            print("invalid input!!")
            continue
        if is_valid_hand(hand, board):
            return hand
        else:
            print("invalid hand!!")


def parse(s: str) -> Tuple[int, int]:
    x, y = s[0], s[1]
    x = ord(x) - ord("a")
    y = int(y) - 1  # to 0-indexed
    # check array index out of bound
    Hand((y, x), Board.init_board())
    return y, x


if __name__ == '__main__':
    main()
