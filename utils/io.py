import cv2
import numpy as np
import open3d as o3d


def load_color(color_fp: str) -> np.ndarray:
    color_suffix = '_rgb.png'
    if not color_fp.endswith(color_suffix):
        color_fp += color_suffix
    color = cv2.imread(color_fp)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return color


def load_depth(depth_fp: str) -> np.ndarray:
    depth_suffix = '_depth.dat'
    if not depth_fp.endswith(depth_suffix):
        depth_fp += depth_suffix
    depth = np.fromfile(depth_fp, dtype=np.int16).reshape(-1, 3)
    return depth


def rgbd_to_pointcloud(color_img: str, depth_img: str) -> o3d.geometry.PointCloud:
    """
    Transforms RGB+D to point cloud, also removing color values
    of rgb = [0, 0, 0], since many of them are invalid
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(depth_img.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(color_img.astype(np.float64).reshape(-1, 3) / 255)
    
    # removing black color
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    non_black_idx = (points[:, 0] != 0) & (points[:, 1] != 0) & (points[:, 2] != 0)
    valid_points = points[non_black_idx].astype(np.float64)
    valid_colors = colors[non_black_idx].astype(np.float64)

    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(valid_colors)
    
    return pcd
