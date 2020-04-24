import torch

from othello.ai import MiniMaxAI, RandomAI, MonteCarloAI, AlphaZero
from othello.helper import extract_valid_hand, put_and_reverse, judge_simple
from othello.model import Board, Hand
from othello.network import Network
from othello.view import view_board


def evaluate(ai1, ai2, sente=True, is_view=False):
    board = Board.init_board()
    if is_view:
        print(view_board(board))
    hands = []
    while True:
        if not extract_valid_hand(board):
            board.side ^= True
            hands.append(Hand.pass_hand())
            print("pass hand!!")
        if not extract_valid_hand(board):
            break
        if board.side is sente:
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
    return judge_simple(board) * (1 if sente else -1)


def main(play_count):
    result = [0, 0, 0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    network = Network(device)
    network.load_state_dict(torch.load('../model/latest.pth'))
    network.eval()
    network.to(device)
    for resnet in network.resnet:
        resnet.to(device)
    ai2 = AlphaZero(network, play_count=50, temperature=1000000, history=True)
    # ai2 = MonteCarloAI(play_count=30)
    for i in range(play_count):
        # ai1 = MonteCarloAI(play_count=50)
        ai1 = RandomAI()
        # ai1 = MonteCarloAI(play_count=30)
        result[evaluate(ai1, ai2, True)] += 1
        if i % (play_count // 10) == 0:
            print(result)
    print("AI 1 win: {}, lose: {}, even: {}".format(result[1], result[-1], result[0]))


if __name__ == '__main__':
    main(100)
    # evaluate(is_view=True)
