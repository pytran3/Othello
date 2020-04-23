from othello.ai import MiniMaxAI, RandomAI, MonteCarloAI, AlphaZero
from othello.helper import extract_valid_hand, put_and_reverse, judge_simple
from othello.model import Board
from othello.network import Network
from othello.view import view_board


def evaluate(ai1, ai2, sente=False, is_view=False):
    board = Board.init_board()
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
        hands.append(hand)
        board = put_and_reverse(hand, board)
        if is_view:
            print(view_board(board))
    # print(ai2.history)
    if is_view:
        print("=" * 10 + "  GAME OVER  " + "=" * 10)
        x_count = (board.board == 1).sum()
        o_count = (board.board == -1).sum()
        print("x: {}, o: {}".format(x_count, o_count))
    return judge_simple(board) * (-1 if sente else 1)


def main(play_count):
    result = [0, 0, 0]
    for i in range(play_count):
        ai1 = RandomAI()
        network = Network()
        ai2 = AlphaZero(network, play_count=20, history=True)
        result[evaluate(ai1, ai2, True)] += 1
        if i % (play_count // 10) == 0:
            print(result)
    print("AI 1 win: {}, lose: {}, even: {}".format(result[1], result[-1], result[0]))


if __name__ == '__main__':
    main(50)
    # evaluate(is_view=True)
