import pickle
from datetime import datetime

import torch

from othello.ai import AlphaZero
from othello.helper import extract_valid_hand, put_and_reverse, judge_simple
from othello.model import Board, Hand
from othello.network import Network

MODEL_PATH = 'model/latest.pth'


def play(network, ai_param=None):
    if ai_param is None:
        ai_param = dict(play_count=100)
    board = Board.init_board()

    ai1 = AlphaZero(network, history=True, **ai_param)
    ai2 = AlphaZero(network, history=True, **ai_param)
    hands = []
    while True:
        if not extract_valid_hand(board):
            board.side ^= True
            hands.append(Hand.pass_hand())
        if not extract_valid_hand(board):
            break
        if board.side:
            hand = ai1.put(board, hands)
        else:
            hand = ai2.put(board, hands)
        hands.append(hand)
        board = put_and_reverse(hand, board)

    result = judge_simple(board)
    history1 = ai1.history if isinstance(ai1, AlphaZero) else []
    history2 = ai2.history if isinstance(ai1, AlphaZero) else []
    for x in history1:
        x[-1] = result
    for x in history2:
        x[0] = x[0] * -1
        x[-1] = -result
    return history1 + history2


def main(play_count):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = Network(device)
    network.load_state_dict(torch.load(MODEL_PATH))
    network.eval()
    network.to(device)
    for resnet in network.resnet:
        resnet.to(device)
    history = []
    for i in range(play_count):
        print(i)
        tmp = play(network)
        history.extend(tmp)
    now = datetime.now()
    path = "data/{:4}{:02}{:02}{:02}{:02}{:02}.history".format(now.year, now.month, now.day, now.hour, now.minute,
                                                               now.second)
    with open(path, "wb") as f:
        pickle.dump(history, f)


if __name__ == '__main__':
    main(5)
