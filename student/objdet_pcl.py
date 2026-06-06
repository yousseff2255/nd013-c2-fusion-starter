import zlib
import cv2
import numpy as np
import torch
import open3d as o3d
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
import misc.objdet_tools as tools


def show_pcl(pcl):
    """Visualize a LiDAR point cloud using Open3D. Press right-arrow to advance frames."""

    def close_callback(vis):
        show_pcl.continue_loop = False

    if not hasattr(show_pcl, "vis"):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Point Cloud Viewer")
        show_pcl.vis = vis
        show_pcl.pcd = o3d.geometry.PointCloud()
        show_pcl.first_frame = True
    else:
        vis = show_pcl.vis

    pcd = show_pcl.pcd
    pcd.points = o3d.utility.Vector3dVector(pcl[:, 0:3])

    if show_pcl.first_frame:
        vis.add_geometry(pcd)
        show_pcl.first_frame = False
    else:
        vis.update_geometry(pcd)

    vis.register_key_callback(262, close_callback)
    show_pcl.continue_loop = True
    while show_pcl.continue_loop:
        vis.update_renderer()
        vis.poll_events()


def show_range_image(frame, lidar_name):
    """Extract and return a stacked range+intensity image from the top LiDAR.
    
    Intensity is normalized between 1st and 99th percentile to suppress outliers.
    """
    for laser in frame.lasers:
        if laser.name == dataset_pb2.LaserName.TOP:
            range_image, _, _ = waymo_utils.parse_range_image_and_camera_projection(laser)

    range_ch  = range_image[:, :, 0]
    intensity = range_image[:, :, 1]

    range_ch[range_ch < 0]   = 0
    intensity[intensity < 0] = 0

    range_ch = (range_ch / np.amax(range_ch) * 255).astype(np.uint8)

    p1, p99 = np.percentile(intensity, 1), np.percentile(intensity, 99)
    intensity = np.clip(intensity, p1, p99)
    intensity = ((intensity - p1) / (p99 - p1) * 255).astype(np.uint8)

    return np.vstack((range_ch, intensity))


def bev_from_pcl(lidar_pcl, configs):
    """Convert a 3D LiDAR point cloud into a 3-channel Bird's-Eye View tensor.
    
    Channels:
        0 — Intensity: normalized surface reflectivity (percentile-clipped)
        1 — Height: top-most z-coordinate per BEV cell, normalized by z-range
        2 — Density: log-normalized point count per BEV cell
    
    Returns a (1, 3, H, W) float tensor ready for the detection backbone.
    """
    # Clip to detection volume
    mask = np.where(
        (lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
        (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
        (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1])
    )
    lidar_pcl = lidar_pcl[mask]
    lidar_pcl[:, 2] -= configs.lim_z[0]  # shift z so ground plane starts at 0

    bev_discretization = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
    pcl = np.copy(lidar_pcl)
    pcl[:, 0] = np.int_((pcl[:, 0] - configs.lim_x[0]) / bev_discretization)
    pcl[:, 1] = np.int_((pcl[:, 1] - configs.lim_y[0]) / bev_discretization)

    # --- Intensity map ---
    # Sort by x, y, then descending z to keep top-most point per cell
    indices = np.lexsort((-pcl[:, 2], pcl[:, 1], pcl[:, 0]))
    pcl = pcl[indices]
    _, unique_idx, counts = np.unique(pcl[:, 0:2], axis=0, return_index=True, return_counts=True)
    pcl_top = pcl[unique_idx]

    p1, p99 = np.percentile(pcl_top[:, 3], 1), np.percentile(pcl_top[:, 3], 99)
    pcl_top[:, 3] = np.clip(pcl_top[:, 3], p1, p99)
    pcl_top[:, 3] = (pcl_top[:, 3] - p1) / (p99 - p1)

    intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    intensity_map[np.int_(pcl_top[:, 0]), np.int_(pcl_top[:, 1])] = pcl_top[:, 3]

    # --- Height map ---
    height_range = configs.lim_z[1] - configs.lim_z[0]
    pcl_top[:, 2] = np.clip(pcl_top[:, 2] / height_range, 0, 1)

    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    height_map[np.int_(pcl_top[:, 0]), np.int_(pcl_top[:, 1])] = pcl_top[:, 2]

    # --- Density map ---
    # Log normalization compresses large counts while preserving relative differences
    normalized_counts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    density_map[np.int_(pcl_top[:, 0]), np.int_(pcl_top[:, 1])] = normalized_counts

    # Assemble 3-channel BEV map and convert to tensor
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[0] = intensity_map[:configs.bev_height, :configs.bev_width]
    bev_map[1] = height_map[:configs.bev_height,    :configs.bev_width]
    bev_map[2] = density_map[:configs.bev_height,   :configs.bev_width]

    bev_tensor = np.zeros((1, *bev_map.shape))
    bev_tensor[0] = bev_map

    return torch.from_numpy(bev_tensor).to(configs.device).float()