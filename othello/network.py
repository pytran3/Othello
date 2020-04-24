from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DN_CHANEL = 128
DN_RESIDUAL_NUM = 3
DN_INPUT_SHAPE = (2, 8, 8)
DN_OUTPUT_SIZE = 64 + 1  # passもある


class Network(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.conv = nn.Conv2d(2, DN_CHANEL, 3, padding=1)
        self.resnet = [ResNet() for _ in range(DN_RESIDUAL_NUM)]
        self.p_clf = nn.Linear(DN_CHANEL, DN_OUTPUT_SIZE)
        self.v_clf = nn.Linear(DN_CHANEL, 1)
        self._device = device

    def predict(self, x) -> Tuple[np.ndarray, float]:
        p, v = self.forward(x)
        p = p.to("cpu").detach().numpy().copy()
        v = v.to("cpu").detach().numpy().copy()
        return p, float(v[0])

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).clone().to(self._device)
        x = self.conv(x)
        for resnet in self.resnet:
            x = resnet(x)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.squeeze(x, -1)
        x = torch.squeeze(x, -1)
        p = self.p_clf(x)
        v = self.v_clf(x)
        p = F.softmax(p, dim=-1)
        v = torch.tanh(v)
        return p, v


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(DN_CHANEL, DN_CHANEL, 3, padding=1)
        self.conv2 = nn.Conv2d(DN_CHANEL, DN_CHANEL, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(DN_CHANEL)
        self.bn2 = nn.BatchNorm2d(DN_CHANEL)

    def forward(self, x):
        sc = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + sc
        x = F.relu(x)
        return x


if __name__ == '__main__':
    # test
    input = np.zeros((2, 2, 8, 8), dtype=np.float32)
    hoge = Network().predict(input)
    print(hoge)
