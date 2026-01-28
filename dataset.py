import os
import math
import shutil
import datetime
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import torch
import torch.utils.data as data
from shapely.geometry import LineString, Polygon, Point
from shapely.affinity import translate
from torch import nn, optim
from torch.backends import cudnn
from tqdm import tqdm


def conv_str_to_dt64(timestr):
    """
    Convert time string to datetime format.
    """
    dt = datetime.datetime.strptime(str(timestr), '%Y%m%d%H')
    dt_str = dt.strftime('%Y-%m-%d %H:00:00')
    return dt_str


def mean_norm(x):
    """
    Apply mean normalization.
    """
    return (x - x.mean()) / x.std()


def raw_data_statistics(file_list, seq_len, noise_level=0.25):
    """
    Calculate statistics for raw data and normalize global XCO2.
    """
    xco2_all = None
    y_all = None  # FFCO2 XCO2 (last column)
    noise_level_percent = int(noise_level * 100)

    for f in tqdm(file_list, desc="Processing raw data"):
        xco2_df = pd.read_csv(f)
        xco2_df["xco2_w_noise"] = xco2_df[f"xco2_w_noise{noise_level_percent}"]
        xco2_df["local_mean"] = xco2_df[f"local_mean{noise_level_percent}"]
        xco2_df["local_std"] = xco2_df[f"local_std{noise_level_percent}"]
        
        # Filter by plume coordinates
        xco2_df = xco2_df[(xco2_df["x_plume"] > -30) & (xco2_df["x_plume"] < 50)]
        xco2_df = xco2_df[(xco2_df["y_plume"] > -40) & (xco2_df["y_plume"] < 40)]
        
        columns = [
            'xco2_w_noise', 'local_mean', 'local_std', 
            'x_plume', 'y_plume', 'ap_wind', 'cp_wind', 'XCO2_A_BV'
        ]
        xco2_np = np.array(xco2_df[columns])
        xco2_x, y = xco2_np[:, :-1], xco2_np[:, -1]

        # Randomly select a sequence of length seq_len
        seq_sel = np.random.choice(xco2_np.shape[0], seq_len)
        xco2_x = xco2_x[seq_sel, :][np.newaxis, :]
        y = y[seq_sel][np.newaxis, :]
        
        if xco2_all is not None:
            xco2_all = np.concatenate((xco2_all, xco2_x), axis=0)
            y_all = np.concatenate((y_all, y), axis=0)
        else:
            xco2_all = xco2_x
            y_all = y  

    # Convert to tensor and calculate statistics
    raw_train = torch.from_numpy(xco2_all).float().reshape(-1, 7)  # (num * seq_len, 7)
    
    mean = torch.mean(raw_train, dim=0, keepdim=True)
    std = torch.std(raw_train, dim=0, keepdim=True)

    # Normalize XCO2 values
    xco2_norm = (xco2_all - mean.numpy()) / std.numpy()
    # Normalize: xco2_value, local_mean, local_std 
    xco2_all[:, :, :3] = xco2_norm[:, :, :3]

    return xco2_all, y_all, mean, std


def get_region_masked(xco2_data, masked_ratio=0.3, group=3):
    """
    Get indices for region-based masking.
    """
    columns = ['xco2_w_noise', 'local_mean', 'local_std', 'x_plume', 'y_plume', 'ap_wind', 'cp_wind']
    df = pd.DataFrame(data=xco2_data, columns=columns)
    center_rows = df.sample(n=group, axis=0) 

    group_masked_num = int(len(df) * masked_ratio / group)
    df['if_selected'] = 0  

    for _, row in center_rows.iterrows():
        center_point = (row['x_plume'], row['y_plume'])
        df['distance'] = np.sqrt((df['x_plume'] - center_point[0])**2 + (df['y_plume'] - center_point[1])**2)
        idx = df[df['if_selected'] == 0].nsmallest(group_masked_num, 'distance').index
        df.loc[idx, 'if_selected'] = 1
    
    region_idx = (df['if_selected'] == 1)
    return region_idx


def get_swath_masked(xco2_data, masked_ratio=0.3, group=3):
    """
    Get indices for swath-based masking (along x or y plume axis).
    """
    columns = ['xco2_w_noise', 'local_mean', 'local_std', 'x_plume', 'y_plume', 'ap_wind', 'cp_wind']
    df = pd.DataFrame(data=xco2_data, columns=columns)
    center_rows = df.sample(n=group, axis=0) 

    axis = np.random.choice(['x_plume', 'y_plume'])
    df['if_selected'] = 0 
    group_masked_num = int(len(df) * masked_ratio / group)

    for _, row in center_rows.iterrows():
        line_value = row[axis]
        df['distance'] = np.abs(df[axis] - line_value)
        idx = df[df['if_selected'] == 0].nsmallest(group_masked_num, 'distance').index
        df.loc[idx, 'if_selected'] = 1

    swath_idx = (df['if_selected'] == 1)
    return swath_idx


class Loader_CO2M(data.Dataset):
    """
    Dataset loader for CO2M data with different masking strategies.
    """
    def __init__(self, raw_train, y, index_list, mask_ratio=0.3):
        self.raw_train = raw_train
        self.y = y
        self.index_list = index_list
        self.mask_ratio = mask_ratio

    def __getitem__(self, index):
        xco2_data = self.raw_train[index]
        y_data = self.y[index]
        xco2_gt = xco2_data.copy()
        
        all_idx = np.arange(len(xco2_data))
        masked_num = int(all_idx.shape[0] * self.mask_ratio)  

        # Randomly choose masking type
        masked_type = np.random.choice(['random', 'region', 'swath'])
        
        if masked_type == 'random':
            masked_idx = np.random.choice(all_idx, size=masked_num, replace=False)
        elif masked_type == 'region':
            masked_idx = get_region_masked(xco2_data, self.mask_ratio, group=3)
        elif masked_type == 'swath':
            # Swath mask: masks a cross/along-plume strip along x/y_plume
            masked_idx = get_swath_masked(xco2_data, self.mask_ratio, group=3) 

        # Mask the first three columns
        xco2_data[masked_idx, :3] = 0

        # Convert to tensors
        xco2_data = torch.from_numpy(xco2_data).float()
        gt_rec = torch.from_numpy(xco2_gt).float()  # Complete features for reconstruction
        gt_y = torch.from_numpy(y_data).float()     # Anthropogenic XCO2 signal

        masked = torch.zeros_like(gt_rec)  # Masked position index
        masked[masked_idx, :3] = 1    
        
        return {
            "xco2_data": xco2_data,
            "gt_rec": gt_rec, 
            "gt_y": gt_y,      # Used for plume regression task (returns bv_ff_xco2)
            "masked": masked   # Indicates masked positions for reconstruction loss
        }
    
    def __len__(self):
        return len(self.index_list)


if __name__ == '__main__':
    # Simple test for masking
    test_xco2_data = np.random.rand(1000, 7)
    test_idx = get_swath_masked(test_xco2_data, masked_ratio=0.3)
    print(f"Masked indices count: {np.sum(test_idx)}")
