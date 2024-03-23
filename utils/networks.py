from copy import deepcopy
import math
import os
import shutil
import struct
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def get_nnmodule_param_count(module: nn.Module):
    param_count = 0
    for param in module.state_dict().values():
        param_count += int(np.prod(param.shape))
    return param_count


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class Sine(nn.Module):
    def __init__(self, w0=30):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        return torch.sin(self.w0 * input)


class SIREN(nn.Module):
    def __init__(
        self,
        coords_channel=3,
        data_channel=1,
        features=256,
        layers=5,
        w0=30,
        output_act=False,
        **kwargs,
    ):
        super().__init__()
        self.net = []
        self.net.append(nn.Sequential(nn.Linear(coords_channel, features), Sine(w0)))
        for i in range(layers - 2):
            self.net.append(nn.Sequential(nn.Linear(features, features), Sine()))
        if output_act:
            self.net.append(nn.Sequential(nn.Linear(features, data_channel), Sine()))
        else:
            self.net.append(nn.Sequential(nn.Linear(features, data_channel)))
        self.net = nn.Sequential(*self.net)
        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)
        # print(self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output

    @staticmethod
    def calc_param_count(coords_channel, data_channel, features, layers, **kwargs):
        param_count = (
            coords_channel * features
            + features
            + (layers - 2) * (features**2 + features)
            + features * data_channel
            + data_channel
        )
        return int(param_count)

    @staticmethod
    def calc_features(param_count, coords_channel, data_channel, layers, **kwargs):
        a = layers - 2
        b = coords_channel + 1 + layers - 2 + data_channel
        c = -param_count + data_channel

        if a == 0:
            features = round(-c / b)
        else:
            features = round((-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a))
        return features


class PosEncodingNeRF(nn.Module):
    def __init__(self, in_channel, frequencies=10):
        super().__init__()

        self.in_channel = in_channel
        self.frequencies = frequencies
        self.out_channel = in_channel + 2 * in_channel * self.frequencies

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_channel)
        coords_pos_enc = coords
        for i in range(self.frequencies):
            for j in range(self.in_channel):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2**i) * math.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2**i) * math.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_channel).squeeze(1)


class Moe(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers):
        super().__init__()

        self.net = []
        self.net.append(nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU()))
        for i in range(layers - 2):
            self.net.append(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()))
        self.net.append(nn.Sequential(nn.Linear(hidden_size, output_size), nn.Softmax(dim=1)))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)