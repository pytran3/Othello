from othello.ai import MiniMaxAI, RandomAI
from othello.helper import extract_valid_hand, put_and_reverse, judge_simple
from othello.model import Board
from othello.view import view_board


def evaluate(sente=False, is_view=False):
    board = Board.init_board()
    board.side = sente
    ai1 = RandomAI()
    ai2 = MiniMaxAI(5)
    if is_view:
        print(view_board(board))
    hands = []
    while True:
        if not extract_valid_hand(board):
            board.side ^= True
        if not extract_valid_hand(board):
            break
        if board.side ^ sente:
            hand = ai1.put(board, hands)
        else:
            hand = ai2.put(board, hands)
        if (board.board == 0).sum() < 12:
            # 計算時間に余裕があるのでdeepに読む
            ai2.depth = 8
        hands.append(hand)
        board = put_and_reverse(hand, board)
        if is_view:
            print(view_board(board))


    if is_view:
        print("=" * 10 + "  GAME OVER  " + "=" * 10)
        x_count = (board.board == 1).sum()
        o_count = (board.board == -1).sum()
        print("x: {}, o: {}".format(x_count, o_count))
    return judge_simple(board) * (-1 if sente else 1)


def main(play_count):
    result = [0, 0, 0]
    for i in range(play_count):
        result[evaluate(i % 2 == 0)] += 1
        if i % (play_count // 10) == 0:
            print(result)
    print("AI 1 win: {}, lose: {}, even: {}".format(result[1], result[-1], result[0]))


if __name__ == '__main__':
    main(50)
    # evaluate(is_view=True)
