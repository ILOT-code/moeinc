import csv
from dataclasses import dataclass
from datetime import datetime
import math
from tqdm import tqdm
import os
import nibabel as nib
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
import argparse
import time
import torch.nn as nn
from einops import rearrange
import numpy as np
import torch.nn.functional as F
from omegaconf import OmegaConf
import tifffile
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import *
from utils.metrics import (
    calc_psnr, 
    calc_ssim, 
    get_folder_size, 
    parse_checkpoints,
)

from utils.networks import (
    SIREN,
    Moeincnet,
    configure_lr_scheduler,
    configure_optimizer,
    get_nnmodule_param_count,
    calc_mlp_param_count,
    calc_mlp_features,
    l2_loss,
    # load_model,
    # save_model,
)
from utils.samplers import RandomPointSampler3D_context1d, RandomPointSampler3D_maskconv,RandomPointSampler3D_maskconv_train


EXPERIMENTAL_CONDITIONS = ["data_name", "data_type", "data_shape", "actual_ratio"]
METRICS = [
    "psnr",
    "ssim",
    "compression_time_seconds",
    "decompression_time_seconds",
    "original_data_path",
    "decompressed_data_path",
]
EXPERIMENTAL_RESULTS_KEYS = (
    ["algorithm_name", "exp_time"] + EXPERIMENTAL_CONDITIONS + METRICS + ["config_path"]
)
timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S.%f")[:-3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="single task for Moeinc mdeical picture compression")
    parser.add_argument(
        "-c",
        type=str,
        default=opj(opd(__file__), "config", "test.yaml"),
        help="yaml file path",
    )
    parser.add_argument("-g", type=str, default="0,1", help="gpu index")
    args = parser.parse_args()

    config_path = os.path.abspath(args.c)
    # Make the gpu index used by CUDA_VISIBLE_DEVICES consistent with the gpu index shown in nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Specify the gpu index to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = args.g
    print(torch.cuda.is_available())
    ###########################
    # 1. load config
    config = OmegaConf.load(config_path)
    output_dir = opj(opd(__file__), "outputs", config.output_dir_name + timestamp)
    os.makedirs(output_dir)
    print(f"All results wll be saved in {output_dir}")
    OmegaConf.save(config, opj(output_dir, "config.yaml"))
    reproduc(config.seed)
    n_training_samples_upper_limit = config.n_training_samples_upper_limit
    n_random_training_samples_percent = config.n_random_training_samples_percent
    n_training_steps = config.n_training_steps
    tblogger = SummaryWriter(output_dir)
    ###########################
    # 2. prepare data, weight_map
    sideinfos = SideInfos3D()
    # parse name and extension
    data_path = config.data.path
    data_name = ops(opb(data_path))[0]
    data_extension = ops(opb(data_path))[-1]
    # read original data
    data = tifffile.imread(data_path)
    # niiimage = nib.load(data_path)
    # data = np.array(niiimage.dataobj).squeeze()
    # print(f"Data shape:{data.shape}")
    if len(data.shape) == 3:
        data = data[..., None]
    assert (
        len(data.shape) == 4
    ), "Only DHWC data is allowed. Current data shape is {}.".format(data.shape)
    data_shape = ",".join([str(i) for i in data.shape])
    sideinfos.depth, sideinfos.height, sideinfos.width, _ = data.shape
    n_samples = sideinfos.depth * sideinfos.width * sideinfos.height
    # denoise data
    denoised_data = denoise(data, config.data.denoise_level, config.data.denoise_close)
    tifffile.imwrite(
        opj(output_dir, data_name + "_denoised" + data_extension),
        denoised_data,
    )
    # normalize data
    sideinfos.normalized_min = config.data.normalized_min
    sideinfos.normalized_max = config.data.normalized_max
    normalized_data = normalize(denoised_data, sideinfos)
    print(f"Memory allocated:{torch.cuda.memory_allocated()/1024/1024}MB")
    # move data to device
    normalized_data = torch.tensor(normalized_data, dtype=torch.float, device="cuda")
    # generate weight_map
    weight_map = generate_weight_map(denoised_data, config.data.weight_map_rules)
    # move weight_map to device
    weight_map = torch.tensor(weight_map, dtype=torch.float, device="cuda")
    ###########################
    # 3. prepare network
    # calculate network structure

    ###########################
    # 4. prepare coordinates
    # shape:(d*h*w,3)
    coord_normalized_min = config.coord_normalized_min
    coord_normalized_max = config.coord_normalized_max
    coordinates = torch.stack(
        torch.meshgrid(
            torch.linspace(coord_normalized_min, coord_normalized_max, sideinfos.depth),
            torch.linspace(
                coord_normalized_min, coord_normalized_max, sideinfos.height
            ),
            torch.linspace(coord_normalized_min, coord_normalized_max, sideinfos.width),
            indexing="ij",
        ),
        axis=-1,
    )
    coordinates = coordinates.cuda()
    ###########################
    print(f"Memory allocated:{torch.cuda.memory_allocated()/1024/1024}MB")

    ###########################
    # 6. prepare sampler
    sampling_required = True
    if n_random_training_samples_percent == 0:
        if n_samples <= n_training_samples_upper_limit:
            sampling_required = False
        else:
            sampling_required = True
            n_random_training_samples = int(n_training_samples_upper_limit)
    else:
        sampling_required = True
        n_random_training_samples = int(
            min(
                n_training_samples_upper_limit,
                n_random_training_samples_percent * n_samples,
            )
        )
    if sampling_required:
        sampler = RandomPointSampler3D_maskconv(coordinates, normalized_data,weight_map,n_random_training_samples,1)
    else:
        sampler = RandomPointSampler3D_maskconv(coordinates, normalized_data,weight_map,0,1)
        coords_batch = sampler.flattened_coordinates
        gt_batch = sampler.flattened_data
        weight_map_batch = sampler.flattened_weight_map
        # coords_batch = rearrange(coordinates, "d h w c-> (d h w) c")
        # gt_batch = rearrange(normalized_data, "d h w c-> (d h w) c")
        # weight_map_batch = rearrange(weight_map, "d h w c-> (d h w) c")
    if sampling_required:
        print(f"Use mini-batch training with batch-size={n_random_training_samples}")
    else:
        print(f"Use batch training with batch-size={n_samples}")
    
    ideal_network_size_bytes = os.path.getsize(data_path) / config.compression_ratio
    ideal_network_parameters_count = ideal_network_size_bytes / 4.0
    print(f"ideal_network_parameters_count:{ideal_network_parameters_count}")

    network = Moeincnet(sampler.context_dim,ideal_network_parameters_count,**config.network_structure)
    actual_network_size_bytes = network.actual_param_count * 4.0
    print(network)
    # 5. prepare optimizer lr_scheduler
    optimizer = configure_optimizer(network.parameters(), config.optimizer)
    lr_scheduler = configure_lr_scheduler(optimizer, config.lr_scheduler)
    # (optional) load pretrainedkf network
    # if config.pretrained_network_path is not None:
    #     load_model(network, config.pretrained_network_path, "cuda")
    # move network to device
    network.cuda()
    # 打印显存占用情况
    print(f"Memory allocated:{torch.cuda.memory_allocated()/1024/1024}MB")
    ###########################
    # 7. optimizing
    checkpoints = parse_checkpoints(config.checkpoints, n_training_steps)
    n_print_loss_interval = config.n_print_loss_interval
    print(f"Beginning optimization with {n_training_steps} training steps.")
    # pbar = trange(1, n_training_steps + 1, desc="Compressing", file=sys.stdout)
    compression_time_seconds = 0
    compression_time_start = time.time()
    for steps in range(1, n_training_steps + 1):
        if sampling_required:
            coords_batch, gt_batch, weight_map_batch,context_batch = sampler.next()
        optimizer.zero_grad()
        # print(torch.cuda.memory_allocated()/1024/1024)
        predicted_batch = network(coords_batch,context_batch)
        # print(torch.cuda.memory_allocated()/1024/1024)
        
        loss_d = l2_loss(predicted_batch, gt_batch, weight_map_batch)
        # loss_d_detach = loss_d.detach()
        # loss_r_detach = network.batchloss.detach()
        labmda = torch.tensor(2, device="cuda")
        # if loss_r_detach * labmda > loss_d_detach:
        #     labmda = loss_d_detach / loss_r_detach
        loss_r = labmda * network.batchloss
        loss = loss_d + loss_r
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if steps % n_print_loss_interval == 0:
            print(f"Memory allocated:{torch.cuda.memory_allocated()/1024/1024}MB")
            compression_time_end = time.time()
            compression_time_seconds += compression_time_end - compression_time_start
            # pbar.set_postfix_str("loss={:.6f}".format(loss.item()))
            print(
                f"#Steps:{steps} Loss:{loss.item()} ElapsedTime:{compression_time_seconds}s ra:{loss_r/loss}",
                
                flush=True,
            )
            tblogger.add_scalar("loss", loss.item(), steps)
            compression_time_start = time.time()
        if steps in checkpoints:
            compression_time_end = time.time()
            compression_time_seconds += compression_time_end - compression_time_start
            # save network and evaluate performance
            curr_steps_dir = opj(output_dir, "checkpoints", f"steps_{steps}")
            os.makedirs(curr_steps_dir)
            compressed_data_save_dir = opj(curr_steps_dir, "compressed")
            os.makedirs(compressed_data_save_dir)
            network_parameters_save_dir = opj(
                compressed_data_save_dir, "network_parameters"
            )
            sideinfos_save_path = opj(compressed_data_save_dir, "sideinfos.yaml")
            OmegaConf.save(sideinfos.__dict__, sideinfos_save_path)
            # save_model(network, network_parameters_save_dir, "cuda")
            # decompress data
            with torch.no_grad():
                d,h,w = sideinfos.depth, sideinfos.height, sideinfos.width
                decompressed_data = torch.zeros(
                    (d,h,w,1),
                    device="cuda",
                )
                dn, hn,wn =2,2,2
                nd, nh, nw = d//dn, h//hn, w//wn
                network.eval()
                decompression_time_start = time.time()
                
                for k in range(0,dn):
                    for i in range(0,hn):
                        for j in range(0,wn):
                            part_coords = coordinates[k*nd:(k+1)*nd, i*nh:(i+1)*nh, j*nw:(j+1)*nw]
                            part_data = decompressed_data[k*nd:(k+1)*nd, i*nh:(i+1)*nh, j*nw:(j+1)*nw]
                            inf_sampler = RandomPointSampler3D_maskconv_train(part_coords, part_data,1)
                            coords0,coords1 = inf_sampler.get_coords()
                            
                            pbar = tqdm(range(100))
                            for _ in pbar:
                                pbar.set_description(f"decompressing the {k},{i},{j}")              
                                context0,_= inf_sampler.get_context()
                                inf_data0 = network(coords0,context0)
                                inf_sampler.set_data(inf_data0,0)
                                
                                _,context1 = inf_sampler.get_context()
                                inf_data1 = network(coords1,context1)
                                inf_sampler.set_data(inf_data1,1)

                            decompressed_data[k*nd:(k+1)*nd, i*nh:(i+1)*nh, j*nw:(j+1)*nw] = inf_sampler.data
                decompression_time_end = time.time()
                decompression_time_seconds = (
                    decompression_time_end - decompression_time_start
                )

                decompressed_data = decompressed_data.cpu().numpy()
                decompressed_data = inv_normalize(decompressed_data, sideinfos)
            # save decompressed data
            decompressed_data_save_dir = opj(curr_steps_dir, "decompressed")
            os.makedirs(decompressed_data_save_dir)
            decompressed_data_save_path = opj(
                decompressed_data_save_dir,
                data_name + "_decompressed" + data_extension,
            )
            # tifffile.imwrite(decompressed_data_save_path, decompressed_data)
            # calculate metrics
            psnr = calc_psnr(data[..., 0], decompressed_data[..., 0])
            ssim = calc_ssim(data[..., 0], decompressed_data[..., 0])
            # record results
            results = {k: None for k in EXPERIMENTAL_RESULTS_KEYS}
            results["algorithm_name"] = "SCI"
            results["exp_time"] = timestamp
            results["original_data_path"] = data_path
            results["config_path"] = config_path
            results["decompressed_data_path"] = decompressed_data_save_path
            results["data_name"] = data_name
            results["data_type"] = config.data.get("type")
            results["data_shape"] = data_shape
            # results["actual_ratio"] = os.path.getsize(data_path) / get_folder_size(
            #     network_parameters_save_dir
            # )
            results["actual_ratio"] = os.path.getsize(data_path)/(network.actual_param_count*4)
            results["psnr"] = psnr
            results["ssim"] = ssim
            results["compression_time_seconds"] = compression_time_seconds
            results["decompression_time_seconds"] = decompression_time_seconds
            csv_path = os.path.join(output_dir, "results.csv")
            if not os.path.exists(csv_path):
                f = open(csv_path, "a")
                csv_writer = csv.writer(f, dialect="excel")
                csv_writer.writerow(results.keys())
            row = [results[key] for key in results.keys()]
            csv_writer.writerow(row)
            f.flush()
            compression_time_start = time.time()
    print("Finish!", flush=True)
