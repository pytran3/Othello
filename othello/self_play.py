import pickle
from datetime import datetime

import torch

from othello.ai import AlphaZero
from othello.helper import extract_valid_hand, put_and_reverse, judge_simple
from othello.model import Board
from othello.network import Network

MODEL_PATH = 'model/latest.pth'


def play():
    board = Board.init_board()
    network = Network()
    network.load_state_dict(torch.load(MODEL_PATH))
    ai1 = AlphaZero(network, exhaust_threshold=8, play_count=20, history=True)
    ai2 = AlphaZero(network, exhaust_threshold=8, play_count=20, history=True)
    hands = []
    while True:
        if not extract_valid_hand(board):
            board.side ^= True
        if not extract_valid_hand(board):
            break
        if board.side:
            hand = ai1.put(board, hands)
        else:
            hand = ai2.put(board, hands)
        hands.append(hand)
        board = put_and_reverse(hand, board)

    result = judge_simple(board)
    history1 = ai1.history
    history2 = ai2.history
    for x in history1:
        x[-1] = result
    for x in history2:
        x[-1] = -result
    return history1 + history2


def main(play_count):
    history = []
    for i in range(play_count):
        print(i)
        tmp = play()
        history.extend(tmp)
    now = datetime.now()
    path = "data/{:4}{:02}{:02}{:02}{:02}{:02}.history".format(now.year, now.month, now.day, now.hour, now.minute,
                                                               now.second)
    with open(path, "wb") as f:
        pickle.dump(history, f)


if __name__ == '__main__':
    main(2)
