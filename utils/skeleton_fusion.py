import os
import copy
import open3d as o3d
import numpy as np
import pandas as pd
from typing import List
import sys
import matplotlib.pyplot as plt

plt.style.use('ggplot')
np.set_printoptions(suppress=True)
sys.path.append('..')

from utils.processing import synchronize_filenames, synchronize_joints, select_points_randomly, distance_between_joints
from utils.visualization import visualize_skeletons_2d
from utils.skeleton_fusion import (
    compute_centroid, compute_threshold, distance_joints_to_centroid,
    distance_all_joints_to_central_joint, normalized_weight_of_frame
)

from PyMoCapViewer import MoCapViewer
from itertools import combinations

import logging
logging.disable(logging.CRITICAL)


def fuse_skeletons_gradient(
    skeletons: List[np.ndarray],
    alpha: float = 1.4,
    beta: float = 1.4
    ) -> np.ndarray:
    '''
    Apply skeleton fusion using a gradient-based weighted average, 
    along with centroid-based distances

    Let p1,...,pK be a joint for camera k=1,..,K
    The fused skeleton p* is given by

    p* = (w1*p1 + w2*p2 + ... + wK*pK)/(w1+...+wK)
    where wi = 1/||pi(t) - pi(t-1)||^alpha * 1/||pi(t) - c(t)||^beta

    with c(t) being the centroid of joints p1,...,pK at time t

    if alpha != 0, it becomes a gradient-based fusion
    if beta != 0, it becomes  centroid-based fusion
    if alpha = beta = 0, then it becomes a simple averaging
    '''
    # TODO: Incorporate m cameras instead of fixing 3
    num_cameras, num_frames, num_joints, num_dims = skeletons.shape
    initial_frame = 20
    fused_joints = np.zeros_like(skeletons[0])
    fused_joints[:initial_frame, :, :] = np.mean(skeletons[:, :initial_frame, :, :], axis=0)

    grads1 = []
    grads2 = []
    grads3 = []

    for f in range(initial_frame, num_frames):
        for j in range(num_joints):
            
            last_point = fused_joints[f - 1, j]

            p1 = skeletons[0, f, j]
            p2 = skeletons[1, f, j]
            p3 = skeletons[2, f, j]
            centroid = (p1 + p2 + p3)/3

            # Distance with respect to last averaged point
            grad1 = np.linalg.norm(p1 - last_point)
            grad2 = np.linalg.norm(p2 - last_point)
            grad3 = np.linalg.norm(p3 - last_point)

            w1 = 1.0 / ((grad1 ** alpha)*(np.linalg.norm(p1 - centroid) ** beta))
            w2 = 1.0 / ((grad2 ** alpha)*(np.linalg.norm(p2 - centroid) ** beta))
            w3 = 1.0 / ((grad3 ** alpha)*(np.linalg.norm(p3 - centroid) ** beta))

            weighted_average = (w1 * p1 + w2 * p2 + w3 * p3)/(w1 + w2 + w3)
            fused_joints[f, j] = weighted_average

    return fused_joints, grads1, grads2, grads3


if __name__ == '__main__':

    master_root_dir = 'C:/Users/Tibor/Documents/Kinect_Data/patient_11/master_1/'
    sub_1_root_dir = 'C:/Users/Tibor/Documents/Kinect_Data/patient_11/sub_1/'
    sub_2_root_dir = 'C:/Users/Tibor/Documents/Kinect_Data/patient_11/sub_2/'
    root_dirs = [master_root_dir, sub_1_root_dir, sub_2_root_dir]
    trafos = [np.load(os.path.join(root_dirs[0], f'transformation_master_sub_{i}.npy')) for i in range(1, len(root_dirs))]
    joint_names = ['SPINE_NAVEL', 'SPINE_CHEST', 'PELVIS', 'HIP_LEFT', 'KNEE_LEFT','ANKLE_LEFT','FOOT_LEFT','HIP_RIGHT','KNEE_RIGHT','ANKLE_RIGHT','FOOT_RIGHT']

    device_idx = 0
    skeleton_dfs, confidence_dfs = synchronize_joints(root_dirs, trafos, joint_names, True)
    pred_dfs = copy.deepcopy(skeleton_dfs[device_idx])
    pcd_filenames = synchronize_filenames(root_dirs).dropna().astype(int).iloc[1:]

    reshaped_skeletons = np.array([skeleton_dfs[i].values.reshape(skeleton_dfs[i].shape[0], -1, 3) for i in range(len(skeleton_dfs))])
    fused_skel, grads1, grads2, grads3 = fuse_skeletons_gradient(reshaped_skeletons, 1.4)
    fused_skel = fused_skel.reshape(fused_skel.shape[0], fused_skel.shape[1]*fused_skel.shape[2])
    fused_df = pd.DataFrame(fused_skel, columns=skeleton_dfs[0].columns)

    for i in range(len(grads1)):
        print(f'grads1: {grads1[i]}, grads2: {grads2[i]}, grads3: {grads3[i]} \n')
    start_f = 100
    end_f = 500

    load_pcds = True

    if load_pcds:
        pcds = []

        for timestamp in pcd_filenames['master_1'].iloc[start_f:end_f+1]:
            master_pcd_fn = os.path.join(master_root_dir, 'filtered_and_registered_pointclouds', str(timestamp) + '.pcd')
            master_pcd = o3d.io.read_point_cloud(master_pcd_fn)

            pcd = copy.deepcopy(master_pcd)
            # points = select_points_randomly(master_pcd, 4096//4)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)
            pcds.append(pcd)
            
        
    viewer = MoCapViewer(grid_axis=None)
    # plotting the fused version
    viewer.add_point_cloud_animation(pcds)
    viewer.add_skeleton(skeleton_dfs[0].iloc[start_f:end_f], color='green')
    viewer.add_skeleton(skeleton_dfs[1].iloc[start_f:end_f], color='green')
    viewer.add_skeleton(skeleton_dfs[2].iloc[start_f:end_f], color='green')
    viewer.add_skeleton(fused_df.iloc[start_f:end_f], color='red')
    viewer.show_window()
