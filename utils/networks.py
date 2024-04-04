from copy import deepcopy
import math
import os
import shutil
import struct
import numpy as np
import torch
from typing import List, Union
from torch import nn
import torch.nn.functional as F
from torch.nn import init

def get_nnmodule_param_count(module: nn.Module):
    param_count = 0
    for param in module.state_dict().values():
        param_count += int(np.prod(param.shape))
    return param_count

def calc_mlp_param_count(input_size, hidden_size, output_size, layers):
    if layers == 1:
        param_count = input_size * output_size + output_size
        return int(param_count)
    param_count = (
        input_size * hidden_size
        + hidden_size
        + (layers - 2) * (hidden_size**2 + hidden_size)
        + hidden_size * output_size
        + output_size
    )
    return int(param_count)

def calc_mlp_features(param_count, input_size, output_size:Union[bool,int], layers):
    # 如果output_size是bool类型，且为False,则输出层大小等于features
    if output_size == False:
        a = layers - 1
        b = input_size + 1 + layers - 1
        c = -param_count
    else:
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

def kaiming_init_weights(m):
    if isinstance (m, (nn.Linear)): 
        init.kaiming_normal_(m.weight)
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
        # # 引入残差连接
        # for i, layer in enumerate(self.net):
        #     if i==0:
        #         output = layer(coords)
        #         outputbackup = output
        #     elif i==len(self.net)-1:
        #         output = layer(output + outputbackup) 
        #     else:
        #         output = layer(output)

        return self.net(coords)


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

class Tok_k(nn.Module):
    def __init__(self, k):
        super(Tok_k, self).__init__()
        self.k = k

    def forward(self, x):
        # 找到前k大的数
        topk_values, topk_indices = torch.topk(x, self.k)
        
        # 把前k大的数保持不变，其它数变为负无穷
        mask = torch.zeros_like(x).scatter(1, topk_indices, 1)
        masked_x = torch.where(mask.bool(), x, torch.full_like(x, float('-inf')))
        
        return masked_x

class Moe(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers):
        super().__init__()

        self.net = []
        self.net.append(nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU()))
        for i in range(layers - 2):
            self.net.append(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()))
        self.net.append(nn.Sequential(nn.Linear(hidden_size, output_size),Tok_k(1),nn.Softmax(dim=-1)))
        self.net = nn.Sequential(*self.net)
        # 初始化暂时按照sine_init
        self.net.apply(sine_init)
        # print(self.net)

    def forward(self, x):
        output = self.net(x)
        return output
#noisy top-k gating
class NoisyTopkRouter(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts,layers=2, top_k=2):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k

        self.net = []
        self.net.append(nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU()))
        for i in range(layers - 2):
            self.net.append(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()))
        self.net.append(nn.Linear(hidden_size, num_experts))
        self.net = nn.Sequential(*self.net)
        self.noise_linear =nn.Linear(input_size, num_experts)

        self.net.apply(kaiming_init_weights)
        self.noise_linear.apply(kaiming_init_weights)
    
    def forward(self, x):
        assert x.dim() == 2, "NoisyTopkPouter Input must have 2 dimensions: B, C"
        logits = self.net(x)
        if self.training:
            noise_logits = self.noise_linear(x)
            #添加噪声
            noise = torch.randn_like(logits)*F.softplus(noise_logits)
            noisy_logits = logits + noise
        else:
            noisy_logits = logits
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        
        routersum = router_output.sum(dim=0)
        cisum = torch.bincount(indices.view(-1))

        if len(cisum) < len(routersum):
            cisum = F.pad(cisum, (0, len(routersum) - len(cisum)), value=0)
        self.batchloss = torch.sum(cisum*routersum)* self.top_k / x.shape[0]**2
        return router_output, indices

class SparseMoE(nn.Module):
    def __init__(self, total_param, router_ratio,
                 router_insize, num_experts, router_layers,top_k,
                 expert_insize, expert_layers,w0,output_act):
        super(SparseMoE, self).__init__()
        router_param = total_param * router_ratio
        expert_param = (total_param - router_param)//num_experts
        router_hidden = calc_mlp_features(router_param, router_insize, num_experts, router_layers)
        self.router = NoisyTopkRouter(router_insize, router_hidden, num_experts, router_layers, top_k)
        expert_hidden = calc_mlp_features(expert_param, expert_insize, False, expert_layers)
        self.decoder = nn.Sequential(nn.Linear(expert_hidden, expert_hidden), nn.ReLU(), nn.Linear(expert_hidden, 1))
        self.experts = nn.ModuleList([SIREN(expert_insize,expert_hidden,expert_hidden,expert_layers,w0,output_act) for _ in range(num_experts)])
        self.top_k = top_k
        self.expert_hidden = expert_hidden
        self.actual_param_count = get_nnmodule_param_count(self.router) + get_nnmodule_param_count(self.experts)

        print(self.router)
        print(self.experts)
    def forward(self, x, context):
        assert x.dim() == 2, "SparseMoE coord Input must have 2 dimensions"
        assert context.dim() == 2, "SparseMoE context Input must have 2 dimensions"
        # print(x.shape)
        # print(context.shape)
        gating_output, indices = self.router(context)
        final_output = torch.zeros((x.size(0),self.expert_hidden)).to(x.device)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)

        return self.decoder(final_output)
    
class conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv.apply(kaiming_init_weights)

    def forward(self, x):
        return self.conv(x)
# self-attention head
class Head(nn.Module):

    def __init__(self, input_size, head_size, context_num, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(input_size, head_size)
        self.query = nn.Linear(input_size, head_size)
        self.value = nn.Linear(input_size, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(context_num, context_num)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    

    
# Multi-Headed Self Attention
class MultiHeadAttention(nn.Module):

    def __init__(self, input_size,num_heads, head_size,context_num, feature_size,dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Head(input_size,head_size,context_num,dropout) for _ in range(num_heads)])
        n_embed = num_heads * head_size
        self.proj = nn.Linear(n_embed*context_num, feature_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        assert x.dim() == 3, "MultiHeadAtten: Input must have 3 dimensions: B, T, C"
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = out.view(out.size(0), -1)
        out = self.dropout(self.proj(out))
        assert out.dim() == 2, "MultiHeadAtten: Output must have 2 dimensions: B*C, F"
        return out
    
class AttentionMoe(nn.Module):
    def __init__(self, coord_size,total_param,router_ratio,output_size,
                 context_dim, num_heads, head_size, context_num, attention_feauture,dropout,
                 top_k, num_experts, router_layers,
                 expert_layers, w0, output_act):
        
        super(AttentionMoe, self).__init__()
        
        self.attention = MultiHeadAttention(context_dim,num_heads,head_size,context_num,attention_feauture,dropout)
        
        assert top_k <= num_experts, "top_k should be less than or equal to num_experts"
        
        self.moe = SparseMoE(total_param,router_ratio,attention_feauture,num_experts,router_layers,top_k,coord_size,expert_layers,w0,output_act)
        self.top_k = top_k
        expert_outsize = self.moe.expert_hidden
        
        self.decoder = nn.Sequential(nn.Linear(expert_outsize, expert_outsize), nn.ReLU(), nn.Linear(expert_outsize,output_size))
        self.decoder.apply(kaiming_init_weights)

    
    def forward(self, x, context):
        context = self.attention(context)
        out = self.moe(x, context)
        out = self.decoder(out)
        return out
    
    def save_atttention(self, path):
        pass
    def save_decoder(self, path):
        pass
    def save_moe(self, path):
        pass
    def load_attention(self, path):
        pass
    def load_decoder(self, path):
        pass
    def load_moe(self, path):
        pass

    
class Moeincnet(nn.Module):
    def __init__(self,context_dim, para_count, moe_ratio, num_sirens, input_size, output_size, layersiren, layersmoe,w0,output_act):
        super().__init__()
        moe_para_count = int(para_count * moe_ratio)
        siren_para_count = para_count - moe_para_count

        moe_features = calc_mlp_features(moe_para_count, context_dim, num_sirens, layersmoe)
        siren_features = calc_mlp_features(int(siren_para_count/num_sirens), input_size, False, layersiren)
        

        self.sirens = nn.ModuleList([SIREN(input_size,siren_features,siren_features,layersiren,w0,output_act) for _ in range(num_sirens)])
        self.moe = NoisyTopkRouter(context_dim, moe_features, num_sirens, layersmoe)
        self.decoder = nn.Sequential(nn.Linear(num_sirens*siren_features, siren_features), nn.ReLU(), nn.Linear(siren_features, output_size))
        self.decoder.apply(sine_init)
        
        self.actual_param_count = get_nnmodule_param_count(self.moe) + get_nnmodule_param_count(self.sirens)
        self.all_param_count = self.actual_param_count + calc_mlp_param_count(num_sirens*siren_features,siren_features,output_size,2)
        self.batchloss = 0

    def forward(self, x, context):
        # print(x.shape)
        siren_outputs = [siren(x) for siren in self.sirens]
        weights,_= self.moe(context)
        # print(weights)
        log_weights = torch.log(weights+1e-10)
        # print(log_weights)
        # print(self.kl_loss)
        self.batchloss = self.moe.batchloss
        weights = weights.repeat_interleave(siren_outputs[0].shape[-1], dim=-1)
        siren_outputs = torch.cat(siren_outputs, dim=-1)
        weight_output = weights * siren_outputs
        return self.decoder(weight_output)


def l2_loss(gt, predicted, weight_map) -> torch.Tensor:
    loss = F.mse_loss(gt, predicted, reduction="none")
    loss = loss * weight_map
    loss = loss.mean()
    return loss


def configure_optimizer(parameters, optimizer_opt) -> torch.optim.Optimizer:
    optimizer_opt = deepcopy(optimizer_opt)
    optimizer_name = optimizer_opt.pop("name")
    if optimizer_name == "Adam":
        Optimizer = torch.optim.Adam(parameters, **optimizer_opt)
    elif optimizer_name == "Adamax":
        Optimizer = torch.optim.Adamax(parameters, **optimizer_opt)
    elif optimizer_name == "SGD":
        Optimizer = torch.optim.SGD(parameters, **optimizer_opt)
    else:
        raise NotImplementedError
    return Optimizer


def configure_lr_scheduler(optimizer, lr_scheduler_opt):
    lr_scheduler_opt = deepcopy(lr_scheduler_opt)
    lr_scheduler_name = lr_scheduler_opt.pop("name")
    if lr_scheduler_name == "MultiStepLR":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, **lr_scheduler_opt
        )
    elif lr_scheduler_name == "CyclicLR":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **lr_scheduler_opt)
    elif lr_scheduler_name == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **lr_scheduler_opt)
    elif lr_scheduler_name == "None":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100000000000]
        )
    else:
        raise NotImplementedError
    return lr_scheduler



# mynet = CombinedNetwork(3, 3, 10, 256, 1, 5, 5)
# print(mynet.parameters)
# x = torch.rand(100,50,3)
# y = mynet(x)
# print(y.shape)

# print(get_nnmodule_param_count(mynet))
# print(calc_mlp_param_count(63,256,3,5)+3*calc_mlp_param_count(63,256,256,5)+calc_mlp_param_count(256*3,256,1,2))

# mynet = Moeincnet(524288,0.3,3,3,2,1,5,2,20)
# print(mynet.actual_param_count)
# print(mynet.all_param_count)
# print(get_nnmodule_param_count(mynet))
# print(mynet.parameters)
# print(mynet.parameters)

# x = torch.tensor([0.0001,0.0003,0.00004])
# x.unsqueeze_(0)
# x1 = torch.tensor([100,1000,100])
# x1.unsqueeze_(0)
# y = mynet(x)
# print(y)
# y = mynet(x1)
# print(y)
# x2 = torch.stack((x,x1),dim=0)
# # print(x)
# y = mynet(x2)
# print(y)

# x = torch.rand(200,3)
# y = mynet(x)
# print(y)