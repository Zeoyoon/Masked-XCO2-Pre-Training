import os
import gc
import logging
import datetime
import pyproj
import numpy as np
import pandas as pd
import torch
from shapely.geometry import LineString, Polygon, Point


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger(filename, verbosity=1, name=None):
    """
    Configure and return a logger instance.
    """
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger


def load_pt_model(model, pt_path):
    """
    Load a pre-trained model checkpoint.
    """
    print(f"=> loading checkpoint '{pt_path}'")
    checkpoint = torch.load(pt_path, map_location='cpu')
    loaded_dict = checkpoint['state_dict']
    sd = model.state_dict()
    
    # Load parameters that match in name and size
    for k in sd:
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
    
    model.load_state_dict(sd) 
    print(f"Loaded checkpoint '{pt_path}' (epoch {checkpoint['epoch']}, best_score {checkpoint['best_score']})")
    
    del loaded_dict
    del sd
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    return model


def lonlat_to_utm_zone(lon, lat):
    """
    Convert longitude and latitude to UTM zone and hemisphere.
    """
    zone_number = int((lon + 180) / 6) + 1
    hemisphere = 'N' if lat >= 0 else 'S'
    return zone_number, hemisphere


def lonlat_to_utmcoord(source_pt, df):
    """
    Convert coordinates from WGS84 to UTM projection.
    """
    # Define projection conversion (from WGS84 to UTM zone)
    zone_number, hemisphere = lonlat_to_utm_zone(source_pt.x, source_pt.y)
    wgs84 = pyproj.CRS("EPSG:4326")
    utm_proj = pyproj.CRS.from_user_input(
        f"EPSG:{32600 + zone_number}" if hemisphere == 'N' else f"EPSG:{32700 + zone_number}"
    )

    proj_transformer_1 = pyproj.Transformer.from_crs(wgs84, utm_proj, always_xy=True)
    
    utm_x_center, utm_y_center = proj_transformer_1.transform(source_pt.x, source_pt.y)
    source_utm = Point((utm_x_center, utm_y_center))
    
    lon_obs = df['longitude'].values
    lat_obs = df['latitude'].values

    df['utm_x'], df['utm_y'] = proj_transformer_1.transform(lon_obs, lat_obs)
    return df, source_utm


def mean_norm(x):
    """
    Standardize data using mean and standard deviation.
    """
    return (x - x.mean()) / x.std()


def transform_distribution(A, B): 
    """
    Transform distribution of A to match that of B.
    """
    A_mean, A_std = np.mean(A), np.std(A)
    B_mean, B_std = np.mean(B), np.std(B)

    A_norm = (A - A_mean) / A_std
    A_new = A_norm * B_std + B_mean
    return A_new


def get_city_radius(ur, epsg_code):
    """
    Calculate the semi-major and semi-minor axes of an urban region.
    """
    # Get simplified boundary coordinates
    ur = ur.simplify(tolerance=0.2, preserve_topology=True)

    if ur.crs.to_epsg() != 4326:
        ur = ur.to_crs(epsg=4326)

    # Re-project to target EPSG code for distance calculations
    ur = ur.to_crs(epsg=epsg_code)

    if ur.geometry[0].geom_type == "Polygon":
        # Select the first Polygon
        ur_geom = ur.geometry[0]
    elif ur.geometry[0].geom_type == "MultiPolygon":
        ur_geom = ur.convex_hull[0]

    area = ur_geom.area
    coords = np.array(ur_geom.exterior.coords)  # Boundary points
    cov = np.cov(coords.T)  # Covariance matrix of points
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort by eigenvalues
    order = np.argsort(eigenvalues)[::-1]
    # Major axis range (approximate length)
    major_axis_range = 2 * np.sqrt(eigenvalues[order[0]])
    # Minor axis range (approximate width)
    minor_axis_range = 2 * np.sqrt(eigenvalues[order[1]])

    # Calculate semi-major and semi-minor axes lengths
    scale_factor = np.sqrt(area / (np.pi * major_axis_range * minor_axis_range))
    semi_major_axis = scale_factor * major_axis_range / 2  # Semi-major axis
    semi_minor_axis = scale_factor * minor_axis_range / 2  # Semi-minor axis

    return semi_major_axis, semi_minor_axis


def conv_str_to_dt64(timestr):
    """
    Convert time string to formatted string.
    """
    dt = datetime.datetime.strptime(str(timestr), '%Y%m%d%H')
    dt_str = dt.strftime('%Y-%m-%d %H:00:00')
    return dt_str
