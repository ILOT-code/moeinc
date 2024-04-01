from typing import Tuple
from einops import rearrange
import numpy as np
import torch


class RandomPointSampler3D:
    def __init__(
        self,
        coordinates: torch.Tensor,
        data: torch.Tensor,
        weight_map: torch.Tensor,
        n_points_per_sampling: int,
    ) -> None:
        self.n_points_per_sampling = n_points_per_sampling
        self.flattened_coordinates = rearrange(
            coordinates, "d h w c-> (d h w) c")
        self.flattened_data = rearrange(data, "d h w c-> (d h w) c")
        self.flattened_weight_map = rearrange(
            weight_map, "d h w c-> (d h w) c")
        self.n_total_points = self.flattened_data.shape[0]

    def next(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled_idxs = torch.randint(
            0, self.n_total_points, (self.n_points_per_sampling,), device="cuda"
        )
        sampled_coords = self.flattened_coordinates[sampled_idxs, :]
        sampled_data = self.flattened_data[sampled_idxs, :]
        sampled_weight_map = self.flattened_weight_map[sampled_idxs, :]
        return sampled_coords, sampled_data, sampled_weight_map


class RandomPointSampler3D_context:
    def __init__(self,
                 coordinates: torch.Tensor,
                 data: torch.Tensor,
                 weight_map: torch.Tensor,
                 n_points_per_sampling: int,
                 context_depth: int):
        d, h, w,_= data.shape
        self.n_points_per_sampling = n_points_per_sampling
        self.flattened_weight_map = rearrange(weight_map, "d h w c-> (d h w) c")
        self.flattened_coordinates = rearrange(coordinates, "d h w c-> (d h w) c")
        self.flattened_data = rearrange(data, "d h w c-> (d h w) c")
        self.n_total_points = self.flattened_data.shape[0]
        contexts = []
        for k in range(context_depth):
            for i in range(context_depth-k):
                for j in range(-i, context_depth-k-i):
                    index = k*h*w + i*w+ j
                    if index == 0:
                        continue
                    context_data = self.get_context(self.flattened_data, index).cuda()
                    context_leftpar = context_data - self.get_context(self.flattened_data, index+1).cuda()
                    context_uppar = context_data - self.get_context(self.flattened_data, index+w).cuda()
                    context_depar = context_data - self.get_context(self.flattened_data, index+w*h).cuda()
                    context = torch.cat([context_data, context_leftpar, context_uppar, context_depar], dim=-1).unsqueeze(1)
                    contexts.append(context)
        self.context_num = len(contexts)
        self.context_dim = contexts[0].shape[-1]
        assert self.context_dim == 4, "context_dim should be 4"
        self.contexts = torch.cat(contexts, dim=1)
        # self.flattened_coordinates = torch.cat([self.flattened_coordinates, self.contexts], dim=-1)

    def get_context(self, data, x):
        new_tensor = torch.cat([torch.zeros(x).cuda(), data[:-x].squeeze(1)])
        return new_tensor.unsqueeze(1)

    def next(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled_idxs = torch.randint(
            0, self.n_total_points, (self.n_points_per_sampling,), device="cuda"
        )
        sampled_coords = self.flattened_coordinates[sampled_idxs, :]
        sampled_data = self.flattened_data[sampled_idxs, :]
        sampled_weight_map = self.flattened_weight_map[sampled_idxs, :]
        sampled_context = self.contexts[sampled_idxs, :]
        return sampled_coords, sampled_data, sampled_weight_map, sampled_context
    
