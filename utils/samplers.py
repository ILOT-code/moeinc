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

class RandomPointSampler3d_plane:
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
        self.contexts = []

        pad_data = torch.nn.functional.pad(data.squeeze(-1), (1, 1, 1, 1, context_depth, 0), mode='constant', value=0)
        print(pad_data.shape)
        ds,de,hs,he,ws,we = context_depth, d+context_depth, 1, h+1, 1,w+1
        for i in range(1,context_depth+1):
            for j in range(-1,2):
                for k in range(-1,2):
                    self.contexts.append(pad_data[ds-i:de-i, hs-j:he-j, ws-k:we-k])

        self.context_dim = len(self.contexts)
        self.contexts = torch.stack(self.contexts, dim=-1).reshape(-1, self.context_dim)
        
        # self.flattened_coordinates = torch.cat([self.flattened_coordinates, self.contexts], dim=-1)

    def get_context(self, data, x):
        new_tensor = torch.cat([torch.zeros(x).cuda(), data[:-x].squeeze(1)])
        return new_tensor.unsqueeze(1)

    def next(self):
        sampled_idxs = torch.randint(
            0, self.n_total_points, (self.n_points_per_sampling,), device="cuda"
        )
        sampled_coords = self.flattened_coordinates[sampled_idxs, :]
        sampled_data = self.flattened_data[sampled_idxs, :]
        sampled_weight_map = self.flattened_weight_map[sampled_idxs, :]
        sampled_context = self.contexts[sampled_idxs, :]
        return sampled_coords, sampled_data, sampled_weight_map, sampled_context

class RandomPointSampler3D_context1d:
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
                    context = self.get_context(self.flattened_data, index).cuda()
                    contexts.append(context)
        self.context_dim = len(contexts)
        self.contexts = torch.cat(contexts, dim=-1)
        # self.flattened_coordinates = torch.cat([self.flattened_coordinates, self.contexts], dim=-1)

    def get_context(self, data, x):
        new_tensor = torch.cat([torch.zeros(x).cuda(), data[:-x].squeeze(1)])
        return new_tensor.unsqueeze(1)

    def next(self):
        sampled_idxs = torch.randint(
            0, self.n_total_points, (self.n_points_per_sampling,), device="cuda"
        )
        sampled_coords = self.flattened_coordinates[sampled_idxs, :]
        sampled_data = self.flattened_data[sampled_idxs, :]
        sampled_weight_map = self.flattened_weight_map[sampled_idxs, :]
        sampled_context = self.contexts[sampled_idxs, :]
        return sampled_coords, sampled_data, sampled_weight_map, sampled_context
    
class RandomPointSampler3D_maskconv:
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
        self.contexts = []

        pad_data = torch.nn.functional.pad(data.squeeze(-1), (context_depth,context_depth,context_depth,context_depth, context_depth, context_depth), mode='constant', value=0)
        
        ds,de,hs,he,ws,we = context_depth, d+context_depth, context_depth, h+context_depth, context_depth,w+context_depth
        for k in range(-context_depth, context_depth+1):
            for i in range(-context_depth, context_depth+1):
                for j in range(-context_depth, context_depth+1):
                    if ((abs(i) + abs(j) + abs(k)) % 2) == 0:
                        continue
                    self.contexts.append(pad_data[ds+k:de+k, hs+i:he+i, ws+j:we+j])
                    
        self.context_dim = len(self.contexts)
        self.contexts = torch.stack(self.contexts, dim=-1).reshape(-1, self.context_dim)            
    def next(self):
        sampled_idxs = torch.randint(
            0, self.n_total_points, (self.n_points_per_sampling,), device="cuda"
        )
        sampled_coords = self.flattened_coordinates[sampled_idxs, :]
        sampled_data = self.flattened_data[sampled_idxs, :]
        sampled_weight_map = self.flattened_weight_map[sampled_idxs, :]
        sampled_context = self.contexts[sampled_idxs, :]
        return sampled_coords, sampled_data, sampled_weight_map, sampled_context
            

             
class RandomPointSampler3D_maskconv_train:
    def __init__(self,
                 coordinates: torch.Tensor,
                 data: torch.Tensor,
                 context_depth: int):
        self.data = data
        self.coordinates = coordinates
        self.context_depth = context_depth
        d, h, w, _ = self.coordinates.shape
        ds, hs, ws = torch.meshgrid(torch.arange(d), torch.arange(h), torch.arange(w),indexing='ij')
        self.mask = (hs + ws) % 2 == ds % 2
        
    def get_coords(self):
        coords0 = self.coordinates[self.mask].reshape(-1, 3)
        coords1 = self.coordinates[~self.mask].reshape(-1, 3)
        return coords0, coords1
    def set_data(self,datanew,flag):
        assert datanew.ndim == 2, "data should be 2D"
        if flag == 0:
            self.data[self.mask] = datanew
        if flag == 1:
            self.data[~self.mask] = datanew
    def get_context(self):
        pad_data = torch.nn.functional.pad(self.data.squeeze(-1), (self.context_depth, self.context_depth, self.context_depth, self.context_depth, self.context_depth, self.context_depth), mode='constant', value=0)
        ds, de, hs, he, ws, we = self.context_depth, self.data.shape[0] + self.context_depth, self.context_depth, self.data.shape[1] + self.context_depth, self.context_depth, self.data.shape[2] + self.context_depth
        contexts = []
        for k in range(-self.context_depth, self.context_depth+1):
            for i in range(-self.context_depth, self.context_depth+1):
                for j in range(-self.context_depth, self.context_depth+1):
                    if ((abs(i) + abs(j) + abs(k)) % 2) == 0:
                        continue
                    contexts.append(pad_data[ds+k:de+k, hs+i:he+i, ws+j:we+j])
                    
        context_dim = len(contexts)
        contexts = torch.stack(contexts, dim=-1)
        
        contexts0 = contexts[self.mask].reshape(-1, context_dim)
        contexts1 = contexts[~self.mask].reshape(-1, context_dim)
        return contexts0, contexts1
        