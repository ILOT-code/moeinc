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

def calc_mlp_param_count(input_size, hidden_size, output_size, layers):
    param_count = (
        input_size * hidden_size
        + hidden_size
        + (layers - 2) * (hidden_size**2 + hidden_size)
        + hidden_size * output_size
        + output_size
    )
    return int(param_count)

def calc_mlp_features(param_count, input_size, output_size, layers):
    a = layers - 2
    b = input_size + 1 + layers - 2 + output_size
    c = -param_count + output_size

    if a == 0:
        hidden_size = round(-c / b)
    else:
        hidden_size = round((-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a))
    return hidden_size

def sine_init(m):
    # print(m)
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
        self.net.append(nn.Sequential(nn.Linear(hidden_size, output_size), nn.Softmax(dim=-1)))
        self.net = nn.Sequential(*self.net)
        # 初始化暂时按照sine_init
        self.net.apply(sine_init)
        # print(self.net)

    def forward(self, x):
        return self.net(x)
    
class CombinedNetwork(nn.Module):
    def __init__(self, num_sirens, input_size, frequencies, hidden_size, output_size, layersiren, layersmoe):
        super().__init__()

        self.pos_enc = PosEncodingNeRF(input_size, frequencies)
        input_size = self.pos_enc.out_channel
        self.sirens = nn.ModuleList([SIREN(input_size,hidden_size,hidden_size,layersiren) for _ in range(num_sirens)])
        self.moe = Moe(input_size, hidden_size, num_sirens, layersmoe)
        self.mlp = nn.Sequential(nn.Linear(num_sirens*hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))
        self.mlp.apply(sine_init)
        

    def forward(self, x):
        x = self.pos_enc(x)
        siren_outputs = [siren(x) for siren in self.sirens]
        weights = self.moe(x).repeat_interleave(siren_outputs[0].shape[-1], dim=-1)
        siren_outputs = torch.cat(siren_outputs, dim=-1)
        weight_output = weights * siren_outputs
        return self.mlp(weight_output)
    

# mynet = CombinedNetwork(3, 3, 10, 256, 1, 5, 5)
# print(mynet.parameters)
# x = torch.rand(100,50,3)
# y = mynet(x)
# print(y.shape)

# print(get_nnmodule_param_count(mynet))
# print(calc_mlp_param_count(63,256,3,5)+3*calc_mlp_param_count(63,256,256,5)+calc_mlp_param_count(256*3,256,1,2))
