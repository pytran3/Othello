import pickle
from pathlib import Path

import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from othello.network import Network

MODEL_PATH = "model/latest.pth"


def train(network, device, x, yp, yv, batch_size=128, verbose=False):
    network.train()
    x = torch.tensor(x).to(device)
    yp = torch.tensor(yp).to(device)
    yv = torch.tensor(yv).to(device)
    dataset = TensorDataset(x, yp, yv)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(network.parameters(), lr=0.00001)
    p_criterion = lambda policy, y_policy: torch.sum((-policy *
                                                      (1e-8 + y_policy.float()).float().log()), 1)
    v_criterion = nn.MSELoss()
    for x, yp, yv in data_loader:
        optimizer.zero_grad()
        p, v = network(x)
        p_loss = p_criterion(p, yp)
        v_loss = v_criterion(v, yv)
        (p_loss + v_loss).sum().backward()
        optimizer.step()
        print(p_loss.sum().item(), v_loss.sum().item())


def main():
    data_path = sorted(Path("data").glob("*.history"))[-1]
    with data_path.open("rb") as f:
        data = pickle.load(f)
    x, yp, yv = zip(*data)
    x = np.array([np.array([i == 1, i == -1], dtype=np.float32) for i in x])
    yp = np.array(yp, dtype=np.float32)
    yv = np.array(yv, dtype=np.float32)

    network = Network()
    network.load_state_dict(torch.load(MODEL_PATH))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    network.to(device)
    for resnet in network.resnet:
        # なぜ必要なんだ・・・再帰的にやってくれ
        resnet.to(device)
    train(network, device, x, yp, yv, batch_size=16)


if __name__ == '__main__':
    main()
