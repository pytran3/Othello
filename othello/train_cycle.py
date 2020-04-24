from datetime import datetime

import numpy as np
import torch

from othello.network import Network
from othello.self_play import play
from othello.train import train

MODEL_PATH = 'model/latest.pth'
BEST_MODEL_PATH = 'model/best.pth'

PLAY_COUNT = 200
EPOCH = 5
EVALUATE_PLAY_COUNT = 50

AI_PARAM = {
    "play_count": 50
}


def _gpu_network(model_path, device):
    network = Network(device)
    network.load_state_dict(torch.load(model_path))
    network.eval()
    network.to(device)
    for resnet in network.resnet:
        resnet.to(device)
    return network


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    network = _gpu_network(MODEL_PATH, device)
    network.eval()

    for train_num in range(100):
        start = datetime.now()
        print("{} train {} {}".format("=" * 10, train_num, "=" * 10))
        # play out
        history = []
        for i in range(PLAY_COUNT):
            tmp = play(network, device, AI_PARAM)
            history.extend(tmp)
            if i % 10 == 0:
                print("self play {}, time {}".format(i, (datetime.now() - start).seconds))

        # train
        network.train()
        for i in range(EPOCH):
            x, yp, yv = zip(*history)
            x = np.array([np.array([i == 1, i == -1], dtype=np.float32) for i in x])
            yp = np.array(yp, dtype=np.float32)
            yv = np.array(yv, dtype=np.float32)
            train(network, device, x, yp, yv, verbose=True)

        # evaluate
        # network.eval()
        # result = [0, 0, 0]
        # for i in range(EVALUATE_PLAY_COUNT):
        #     latest_ai = AlphaZero(network, **AI_PARAM)
        #     best_ai = RandomAI()
        #     result[evaluate(latest_ai, best_ai, sente=i % 2 == 0)] += 1
        # # latest
        # win = result[1]
        # lose = result[-1]
        # even = result[0]
        # print("latest model: win={}, lose={}, even={}".format(win, lose, even))
        # if win > lose:
        #     best_network = network
        #     torch.save(best_network.state_dict(), BEST_MODEL_PATH)
        torch.save(network.state_dict(), MODEL_PATH)
        print("train {}  finish. time {}".format(train_num, (datetime.now() - start).seconds))


if __name__ == '__main__':
    main()
