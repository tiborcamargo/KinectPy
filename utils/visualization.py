import open3d as o3d
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Union, List
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
        if type(x) == type(np.array([])):
            points = x[batch_idx, :, :]
        else:
            points = x[batch_idx, :, :].numpy()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0, 0, 0])
        pcds.append(pcd)

    # Visualizer
    viewer = MoCapViewer(grid_axis=None, sampling_frequency=frequency)
    viewer.add_point_cloud_animation(pcds)
    
    # Generate joints
    if type(y_label) != type(None):
        columns = np.array([[f'joint_{i}x', f'joint_{i}y', f'joint_{i}z'] for i in range(y_label.shape[1]//3)]).flatten()
        skeleton_gt = pd.DataFrame(y_label, columns=columns)
        viewer.add_skeleton(skeleton_gt, color='green')

    if type(y_pred) != type(None):
        columns = np.array([[f'joint_{i}x', f'joint_{i}y', f'joint_{i}z'] for i in range(y_pred.shape[1]//3)]).flatten()
        skeleton_pred = pd.DataFrame(y_pred, columns=columns)
        viewer.add_skeleton(skeleton_pred, color='red')
        
    viewer.show_window()


def display_pcd_and_joints(
    pointclouds: Union[None, List[o3d.geometry.PointCloud]] = None, 
    joints: Union[None, np.ndarray] = None,
    joints_columns: List[str] = ['PELVIS']
    ) -> None:
    '''
    Args:
        joints: array of joints for one or multiple devices, 
                it accepts the following shape:
                joints.shape = (num_of_frames, num_of_devices, num_of_joints*3)
                where `num_of_device` expects at least one device (master_1 or sub_1/sub_2/...)
    
    '''

    print('TODO: Implement joint names')
    
    viewer = MoCapViewer(grid_axis=None)
    if pointclouds is not None:
        viewer.add_point_cloud_animation(pointclouds)

    if joints is not None:
        num_frames, num_devices, cols = np.shape(joints)
        for i in range(num_devices):
            if i == 0:
                c = 'red'
            elif i == 1:
                c = 'green'
            elif i == 2:
                c = 'blue'
            else:
                c = 'gray'
            viewer.add_skeleton(
                pd.DataFrame(joints[:, i, :], columns=joints_columns), 
                skeleton_connection='azure', 
                color=c
                )

    viewer.show_window()
