import open3d as o3d
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Union
from PyMoCapViewer import MoCapViewer


def visualize_predictions(
    x: Union[tf.Tensor, np.ndarray], 
    y_label: Union[tf.Tensor, np.ndarray, None] = None,
    y_pred: Union[tf.Tensor, np.ndarray, None] = None,
    frequency: int = 10
    ) -> None:
    """ 
    Given a set of points with shape x.shape (n_batches, m_joints, 3) 
    and its corresponding set of joint poses y.shape (n_batches, m_joints*3)
    returns the skeleton + point cloud visualization 
    """

    # Generate point clouds
    pcds = []
    for batch_idx in range(x.shape[0]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x[batch_idx, :, :].numpy())
        pcd.paint_uniform_color([0, 0, 0])
        pcds.append(pcd)

    # Visualizer
    viewer = MoCapViewer(grid_axis=None, sampling_frequency=frequency)
    viewer.add_point_cloud_animation(pcds)
    
    # Generate joints
    if y_label:
        columns = np.array([[f'joint_{i}x', f'joint_{i}y', f'joint_{i}z'] for i in range(y_label.shape[1]//3)]).flatten()
        skeleton_gt = pd.DataFrame(y_label, columns=columns)
        viewer.add_skeleton(skeleton_gt, color='green')

    if y_pred:
        columns = np.array([[f'joint_{i}x', f'joint_{i}y', f'joint_{i}z'] for i in range(y_pred.shape[1]//3)]).flatten()
        skeleton_pred = pd.DataFrame(y_pred, columns=columns)
        viewer.add_skeleton(skeleton_pred, color='red')
        
    viewer.show_window()
