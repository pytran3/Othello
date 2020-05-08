from datetime import datetime

import numpy as np
import torch

from othello.network import Network
from othello.self_play import play
from othello.train import train

MODEL_PATH = 'model/latest.pth'
DATETIME_MODEL_PATH = 'model/{}{}{}.pth'

PLAY_COUNT = 200
EPOCH = 1000
PATIENCE = 5
AI_PARAM = {
    "play_count": 100
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
        network.eval()
        for i in range(PLAY_COUNT):
            tmp = play(network, AI_PARAM)
            history.extend(tmp)
            if i % 10 == 0:
                print("self play {}, time {}".format(i, (datetime.now() - start).seconds))

        # train
        network.train()
        patience = 0
        best_p_loss = float("inf")
        print('history num:', len(history))
        for i in range(EPOCH):
            if patience > PATIENCE:
                break
            patience += 1
            x, yp, yv = zip(*history)
            x = np.array([np.array([i == 1, i == -1], dtype=np.float32) for i in x])
            yp = np.array(yp, dtype=np.float32)
            yv = np.array(yv, dtype=np.float32)
            p_loss, v_loss = train(network, device, x, yp, yv, verbose=True)
            print(i, p_loss / len(history), v_loss / len(history))
            if p_loss < best_p_loss:
                patience = 0
                best_p_loss = p_loss

        torch.save(network.state_dict(), MODEL_PATH)
        today = datetime.now()
        torch.save(network.state_dict(), DATETIME_MODEL_PATH.format(today.year, today.month, today.day))
        print("train {}  finish. time {}".format(train_num, (datetime.now() - start).seconds))


if __name__ == '__main__':
    main()
