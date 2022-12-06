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

    # sub1 = 0022..., sub2 = 0017... 
    print('Red: master\tGreen: sub 1\tBlue: sub 2')
    viewer.show_window()

def visualize_skeletons_2d(
        skeleton_dataframes: List[pd.DataFrame], 
        start_frame: int = 0, 
        end_frame: int = 10000, 
        skip_every_n_frame: int = 5,
        plane: str = 'xy',
        title: str = '',
        include_frame: bool = False,
        xlim: List[int] = [-950, 1050],
        ylim: List[int] = [700, -1100],
        labels: List[str] = ''
    ) -> None:
    ''' 
    Plotting n skeletons in the 2d selected `plane` 
    where n = len(skeleton_dataframes)
    '''
    import matplotlib.pyplot as plt

    if plane == 'xy':
        coords = [0, 1]
    if plane == 'xz':
        coords = [0, 2]
    if plane == 'yz':
        coords = [1, 2]

    if labels != '':
        assert len(labels) == len(skeleton_dataframes), 'The list of labels must match the number of skeletons'

    end_frame = min(end_frame, len(skeleton_dataframes[0]))
    for frame in np.arange(start_frame, end_frame, skip_every_n_frame):
        plt.figure(figsize=(8, 8))
        plt.grid()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('x-axis (mm)')
        plt.ylabel('y-axis (mm)')
        for device_idx in range(len(skeleton_dataframes)):
            alpha = 0.4
            if device_idx == 0:
                c = 'r'
            if device_idx == 1:
                c = 'b'
            if device_idx == 2:
                c = 'y'
            if device_idx > 2:
                c = 'k'
                alpha = 1
            
            ith_frame = skeleton_dataframes[device_idx].iloc[frame].values.reshape(-1, 3)
            
            x1 = ith_frame[:7, coords[0]]
            x2 = np.concatenate((ith_frame[:3, coords[0]], ith_frame[7:, coords[0]]))
            y1 = ith_frame[:7, coords[1]]
            y2 = np.concatenate((ith_frame[:3, coords[1]], ith_frame[7:, coords[1]]))

            if labels == '':
                label = 'Sensor Idx: ' + str(device_idx)
            else:
                label = labels[device_idx]

            plt.plot(x1, y1, marker='o', label=label, c=c, alpha=alpha)  
            plt.plot(x2, y2, marker='o', c=c, alpha=alpha)  

        fig_title = f'{title}'
        if include_frame:
            fig_title += f' - Frame: {frame}'
        plt.title(fig_title)
        plt.legend()
    plt.show()


def generate_timeseries_plot(
        root_dir: str = 'E:/Extracted_data/p1/normal/il_cmj/',
        joint: str = 'PELVIS',
        alpha: float = 0.5,
        beta: float = 0.5
    ) -> None:
    '''
    Visualize a time-series of each x,y, z axis for a certain joint
    '''
    import os
    import matplotlib.pyplot as plt
    from utils.skeleton_fusion import fuse_skeletons_gradient_centroid

    root_dir = os.path.abspath(root_dir)
    patient = root_dir.split('\\')[2]

    master_1_df = pd.read_csv(os.path.join(root_dir, 'master_1/skeleton/registered_and_synced_positions_3d.csv'),
                              delimiter=';')
    sub_1_df = pd.read_csv(os.path.join(root_dir, 'sub_1/skeleton/registered_and_synced_positions_3d.csv'),
                           delimiter=';')
    sub_2_df = pd.read_csv(os.path.join(root_dir, 'sub_2/skeleton/registered_and_synced_positions_3d.csv'),
                           delimiter=';')

    min_len = min(len(master_1_df), len(sub_1_df), len(sub_2_df))

    master_1_df = master_1_df[[joint + ' (x)', joint + ' (y)', joint + ' (z)']].iloc[:min_len]
    sub_1_df = sub_1_df[[joint + ' (x)', joint + ' (y)', joint + ' (z)']].iloc[:min_len]
    sub_2_df = sub_2_df[[joint + ' (x)', joint + ' (y)', joint + ' (z)']].iloc[:min_len]

    skeleton_dfs = [master_1_df, sub_1_df, sub_2_df]
    skeletons = np.array(
        [skeleton_dfs[i].values.reshape(skeleton_dfs[i].shape[0], -1, 3) for i in range(len(skeleton_dfs))])
    fused_skeleton = fuse_skeletons_gradient_centroid(skeletons, alpha=alpha, beta=beta)[:, 0, :]

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(3, 1, sharex=True, squeeze=True)
    fig.set_size_inches(25, 12)

    ax[0].set_ylabel('x position (mm)');  # ax[0].set_xlabel('frame');
    ax[0].plot(master_1_df[joint + ' (x)'].values, c='g')
    ax[0].plot(sub_1_df[joint + ' (x)'].values, c='b')
    ax[0].plot(sub_2_df[joint + ' (x)'].values, c='k')
    ax[0].plot(fused_skeleton[:, 0], c='r')

    ax[1].set_ylabel('y position (mm)');  # ax[1].set_xlabel('frame');
    ax[1].plot(master_1_df[joint + ' (y)'].values, c='g', label='$C_1$')
    ax[1].plot(sub_1_df[joint + ' (y)'].values, c='b', label='$C_2$')
    ax[1].plot(sub_2_df[joint + ' (y)'].values, c='k', label='$C_3$')
    ax[1].plot(fused_skeleton[:, 1], c='r', label=r'Fusion $\alpha = {}, \beta = {}$'.format(alpha, beta))
    ax[1].legend()

    ax[2].set_ylabel('z position (mm)');
    ax[2].set_xlabel('frame');
    ax[2].plot(master_1_df[joint + ' (z)'].values, c='g')
    ax[2].plot(sub_1_df[joint + ' (z)'].values, c='b')
    ax[2].plot(sub_2_df[joint + ' (z)'].values, c='k')
    ax[2].plot(fused_skeleton[:, 2], c='r')

    fig.suptitle(f'Patient: {patient.upper()} - Joint: {joint}')
    fig.set_tight_layout(tight=True)
    plt.show()