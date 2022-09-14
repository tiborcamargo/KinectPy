import os, sys
sys.path.append('..')
import copy
import numpy as np
import pandas as pd
import open3d as o3d
import logging
import tensorflow as tf
from utils.processing import scale_point_cloud, select_points_randomly, sort_filenames_by_timestamp, normalize_pointcloud, obb_normalization
from utils.visualization import visualize_predictions
from datasets.kinect_dataset import KinectDataset
from scipy.spatial.transform import Rotation as R
from typing import Tuple


def obb_normalization_batch(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """"
    Apply the following normalization method:
        p_obb = p@(R_obb)
        p_nor = (p_obb - p_c)/L_obb
        
    Where p are the points of the point cloud pcd
    R_obb is the rotation matrix from the Oriented Bounding Box (OBB)
    p_c is the center of the OBB
    L_obb is the maximum edge length of the OBB 
    
    Around [-1, 1] but needs proof
    """
    if len(x.shape) == 2:
        x = x[np.newaxis, :]
        
    obb_points = []
    obb_joints = []

    for batch_idx in range(x.shape[0]):

        points = x[batch_idx]
        joints = y[batch_idx]

        # getting OBB normalization parameters
        pcd_tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        obb_tmp = pcd_tmp.get_oriented_bounding_box()
        R_obb = obb_tmp.get_rotation_matrix_from_yxz([0, np.pi, 0]) 
        b_c = obb_tmp.get_center()
        L_obb = np.max(obb_tmp.extent)

        # Normalizing the point cloud
        x_obb = np.matmul(points, R_obb)
        x_nor = (x_obb + b_c)/L_obb
        obb_points.append(x_nor)

        # Normalizing the joints
        y_reshaped = joints.reshape(-1, 3)
        y_obb = np.matmul(y_reshaped, R_obb)
        y_nor = (y_obb + b_c)/L_obb
        y_nor = y_nor.reshape(joints.shape[0])
        obb_joints.append(y_nor)
    
    obb_points = np.array(obb_points)
    obb_joints = np.array(obb_joints)
    
    return obb_points, obb_joints
        

def obb_rotation_translation_batch(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    obb_points = []
    obb_joints = []
    r = R.from_euler('z', 90, degrees=True).as_matrix()

    for batch_idx in range(x.shape[0]):
        points = x[batch_idx]
        joints = y[batch_idx]
        
        # Create temp pcd to retrieve OBB translation + rotation
        tmp_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))

        # Retrieve rotation and translation
        obb = tmp_pcd.get_oriented_bounding_box()
        obb_rot = obb.R
        obb_trans = obb.get_center()

        # Normalize points
        obb_normalized_points = (points - obb_trans)@obb_rot@r
        obb_points.append(obb_normalized_points)

        # Normalize joints
        joints_3d = joints.reshape((joints.shape[0]//3, 3))
        obb_normalized_joints = (joints_3d - obb_trans)@obb_rot@r
        obb_normalized_joints = obb_normalized_joints.reshape((joints.shape[0]))
        obb_joints.append(obb_normalized_joints)

    obb_points = np.array(obb_points)
    obb_joints = np.array(obb_joints)
    
    return obb_points, obb_joints


def translation_normalization_batch(x, y):
    obb_points = []
    obb_joints = []

    for batch_idx in range(x.shape[0]):
        points = x[batch_idx]
        joints = y[batch_idx]
        
        # Create temp pcd to retrieve OBB translation + rotation
        tmp_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))

        # Retrieve rotation and translation
        obb = tmp_pcd.get_oriented_bounding_box()
        obb_trans = obb.get_center()

        # Normalize points
        obb_normalized_points = (points - obb_trans)
        obb_points.append(obb_normalized_points)

        # Normalize joints
        joints_3d = joints.reshape((joints.shape[0]//3, 3))
        obb_normalized_joints = (joints_3d - obb_trans)
        obb_normalized_joints = obb_normalized_joints.reshape((joints.shape[0]))
        obb_joints.append(obb_normalized_joints)

    obb_points = np.array(obb_points)
    obb_joints = np.array(obb_joints)
    
    return obb_points, obb_joints


@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 3)), tf.TensorSpec(shape=(None, None))])
def normalize_obb(x, y):
    x, y = tf.numpy_function(obb_normalization_batch, [x, y], (tf.double, tf.double))
    return x, y


@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 3)), tf.TensorSpec(shape=(None, None))])
def normalize_obb_rotation_translation(x, y):
    x, y = tf.numpy_function(obb_rotation_translation_batch, [x, y], (tf.double, tf.double))
    return x, y


@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 3)), tf.TensorSpec(shape=(None, None))])
def translate(x, y):
    x, y = tf.numpy_function(translation_normalization_batch, [x, y], (tf.double, tf.double))
    return x, y
