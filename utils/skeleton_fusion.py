import os
import copy
import open3d as o3d
import numpy as np
import pandas as pd

from typing import List, Union
from utils.processing import synchronize_filenames, synchronize_joints, select_points_randomly, distance_between_joints
from utils.visualization import visualize_skeletons_2d
from PyMoCapViewer import MoCapViewer
from itertools import combinations


def compute_threshold(
    skeleton_dataframes: List[pd.DataFrame],
    confidence_dataframes: List[pd.DataFrame]
    ) -> float:
    '''
    (Equation 12)
    '''
    num_frames = len(skeleton_dataframes[0])
    sum_of_dists = 0
    
    for frame in range(num_frames):
        sum_of_dists += average_distance_between_joints(
            skeleton_dataframes, 
            confidence_dataframes, 
            frame
        )
        
    alpha = (1/num_frames)*sum_of_dists
    return alpha


def average_distance_between_joints(
    skeleton_dataframes: List[pd.DataFrame], 
    confidence_dataframes: List[pd.DataFrame], 
    frame: int
    ) -> float:
    '''
    (Equation 10)
    '''
    
    average_distance_between_joints = 0
    joints_tracked = joints_tracked_by_all_cameras(confidence_dataframes, frame)
    
    for joint in joints_tracked:
        average_distance_between_joints += distance_between_joints(skeleton_dataframes, frame, joint)

    average_distance_between_joints = average_distance_between_joints/len(joints_tracked)
    return average_distance_between_joints


def joints_tracked_by_all_cameras(
    confidence_dataframes: List[pd.DataFrame], 
    frame: int
    ) -> List[int]:
    '''
    Given C = {C1, ..., Cm} cameras, return which joints are being 
    CONFIDENTLY tracked by ALL cameras in an specific frame.
    
    For example if joints 0, 1, 2 and 5 are being tracked by all
    Cm cameras, then it returns:
         joints_tracked_all_cameras = [0, 1, 2, 5]
    '''
    number_of_cameras = len(confidence_dataframes)
    joints_tracked = []
    
    joints_tracked = []
    for c in range(3):
        tracked_joints_mask = confidence_dataframes[c].iloc[frame] == 2
        tracked = tracked_joints_mask[tracked_joints_mask == True].index
        tracked = np.nonzero(tracked)[0].tolist()
        joints_tracked.append(tracked)

    joints_tracked_all_cameras = list(set.intersection(*map(set, joints_tracked)))
    
    return joints_tracked_all_cameras


def weight_of_frame(confidence_dataframe: pd.DataFrame, frame: int):
    return np.sum(confidence_dataframe.iloc[frame])


def distance_between_joints(
    skeleton_dataframes: List[pd.DataFrame], 
    frame: int, 
    joint: int
    ) -> float:
    ''' 
    (Equation 11)

    Compute the distance between all correspondent joints `joint` and 
    normalize by the number of possible camera combinations. 
    
    For example: 
        joint = 0 ('KNEE_RIGHT')
            dist(knee_right_camera_0, knee_right_camera_1) = x01
            dist(knee_right_camera_0, knee_right_camera_2) = x02
            dist(knee_right_camera_1, knee_right_camera_2) = x12
        return
            total_dist: (x01 + x02 + x12)/C(3, 2)
    '''
    if joint > len(skeleton_dataframes[0].columns)//3 - 1:
        raise ValueError('Not enough joints in the dataframe')
        
    m_cameras = len(skeleton_dataframes)
    camera_combinations = list(combinations(range(m_cameras), 2))
    num_combinations = len(camera_combinations)
    
    total_dist = 0
    for c1, c2 in camera_combinations:
        total_dist += np.linalg.norm(
            skeleton_dfs[c1].iloc[frame][joint*3:(joint+1)*3] - skeleton_dfs[c2].iloc[frame][joint*3:(joint+1)*3]
        )
    total_dist = total_dist/num_combinations
    return total_dist


def compute_centroid(
    joints_per_camera: List[np.ndarray], 
    frame: int,
    joint: int
    ) -> np.ndarray:
    '''
    Compute centroid for the joint `joint` seen by multiple cameras
    Assumes that each element in `joints_per_camera` is reshaped
    in the shape (num of frames, num of joints, 3)
    
    Args:
      joints_per_camera: 
        List containing the joints obtained for each camera
        with the following array shape:
        joints_per_camera[i] : (number of frames, number of joints, 3)
                
    Returns:
      centroids:
        centroid for all the joints in frame `frame`,
        with the following shape:
        centroids = (number of joints, 3)
    '''
    number_of_cameras = len(joints_per_camera)
    theta_i = np.zeros_like(joints_per_camera[0][frame][joint])
    for c in range(number_of_cameras):
        theta_i += joints_per_camera[c][frame][joint]
        
    centroids = theta_i/number_of_cameras
    return centroids


def distance_joints_to_centroid(
    joints_per_camera: List[np.ndarray],
    centroids: np.ndarray,
    camera_idx: int,
    frame: int,
    joint: int
    ) -> np.ndarray:
    ''' 
    Compute the distance of a single joint seen by all camera `camera_idx` 
    and the centroid
    '''
    return np.linalg.norm(reshaped_skeletons[camera_idx][frame][joint] - theta_i) 


def find_center_joint(
    joints_per_camera: List[np.ndarray],
    centroids: np.ndarray,
    camera_idx: int,
    frame: int
    ) -> np.ndarray:
    return 


def distance_all_joints_to_central_joint(
    joints_per_camera: List[np.ndarray],
    central_joint: int,
    frame: int,
    joint: int
    ) -> np.ndarray:
    '''
    Return the distance between the joints seen by each camera and the central joint
    '''
    
    number_of_cameras = len(joints_per_camera)
    dists = []
    
    for c in range(number_of_cameras):
        dist = np.linalg.norm(joints_per_camera[c][frame][joint] - joints_per_camera[central_joint][frame][joint])
        dists.append(dist)
        
    return dists


def placeholder():
    ''' Computes equation (5) in an efficient way, for all cameras and joints '''
    return np.linalg.norm(reshaped_skeletons - theta_i, axis=3)
