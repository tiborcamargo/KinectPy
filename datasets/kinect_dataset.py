import os
import random
import open3d as o3d
import pandas as pd
import tensorflow as tf
import numpy as np
from options.joints import JOINTS_INDICES
from typing import List, Literal
import gc
gc.disable()
np.set_printoptions(suppress=True)


class KinectDataset:
    def __init__(
        self,
        subjects_dirs, 
        joints: List[str],
        number_of_points: int,
        flag: Literal['train', 'val', 'test']
        ):
        """
        Args:
            subjects_dirs: ['/path/to/train/subject1/', '/path/to/train/subject2', ...]
            batch_size: Number of points in a batch
            joints: Skeleton joints defined by your RGB-D sensor, such as: ['PELVIS', 'FOOT', ...],
                    see all options in options folder
            number_of_points: Number of points to be sub-sampled in a point cloud
            flag: whether the dataset is used for training, validation or testing,
                  it is used to enforce a directory structure such as 
                  /path/to/train/... 
        """
        # Dataset size will be increased when calling tf.data.Dataset.from_generator
        self.dataset_size = 0
        self.joints = joints
        self.output_types = (tf.float32, tf.float32)
        self.number_of_joints = len(joints)
        self.number_of_points = number_of_points
        self.output_shapes = ((self.number_of_points, 3), (self.number_of_joints*3))
        self.flag = flag
        self.joints_columns = np.concatenate([[joint + ' (x)', joint + ' (y)', joint + ' (z)']  for joint in joints])

        master_root_dirs = []
        for subject in subjects_dirs:
            for experiment in os.listdir(subject):
                if experiment.find('_') < 0:
                    master_dir = os.path.join(subject, experiment, 'master_1')
                    master_root_dirs.append(master_dir)
                    
        random.shuffle(master_root_dirs)
        self.pointcloud_files = []
        self.correspondent_skeleton_csv = {}

        for master_root_dir in master_root_dirs:
            pcd_dir = os.path.join(master_root_dir, 'filtered_and_registered_pointclouds')
            self.pointcloud_files.extend([os.path.join(pcd_dir, fn) for fn in os.listdir(pcd_dir)])

            skeleton_fp = pcd_dir.replace('filtered_and_registered_pointclouds', os.path.join('skeleton', 'synced_positions_3d.csv'))
            self.correspondent_skeleton_csv[pcd_dir] = pd.read_csv(skeleton_fp, index_col='timestamp')

        self.dataset_size = len(self.pointcloud_files)

        # Creating tensorflow dataset
        self.tf_dataset = tf.data.Dataset.from_generator(
            self._pointcloud_skeleton_tf_generator, 
            args= [self.pointcloud_files, self.number_of_points],
            output_types = self.output_types,
            output_shapes = self.output_shapes
            )
        
        
    def _pointcloud_skeleton_tf_generator(
        self,
        pointcloud_files: List[str], 
        number_of_points: int,
        ):
        """
        Args:
            pointcloud_files: Filepaths of all point clouds used
            number_of_points: Downsampling a pointclod to use *number_of_points*
        """
        pointcloud_files = [file.decode('utf-8') for file in pointcloud_files]
        i = 0 
        for file in pointcloud_files:
            # The try-exception is here for when pcds cant find a correspondence in csv
            try:
                # Finding the proper skeleton dataframe and the proper timestamp 
                timestamp = int(file.split(os.path.sep)[-1][:-4])

                correspondent_skeleton_key = tf.io.gfile.join(
                    os.path.sep.join(file.split(os.path.sep)[:-2]), 
                    'filtered_and_registered_pointclouds'
                )

                # Retrieving skeletons positions and joints
                skeleton_df = self.correspondent_skeleton_csv[correspondent_skeleton_key]
                skeleton_positions = skeleton_df.loc[timestamp][self.joints_columns].values

                ## Reading point cloud and sampling
                pcd = o3d.io.read_point_cloud(file)
                pcd_points = self._select_points_randomly(pcd, number_of_points)

                yield pcd_points, skeleton_positions
                i = i + 1

            except Exception as e:
                print(e, flush=True)
                
                
    def __call__(self):
        return self.tf_dataset#.take(self.dataset_size)
    
    
    def _select_points_randomly(
        self,
        pointcloud: o3d.geometry.PointCloud, 
        number_of_points: int 
        ) -> np.ndarray:
        """
        Because each point cloud contains a large amount of points, we need 
        to sample a smaller subset of points to work with.

        Currently only uses spatial information, but no color information
        """
        pcd_points = np.asarray(pointcloud.points)
        sampled_idx = np.random.choice(np.arange(0, len(pcd_points), 1), 
                                       size=number_of_points, 
                                       replace=False)
        pcd_points = pcd_points[sampled_idx]
        pcd_colors = None
        return pcd_points
    
    
    def visualize_dataset(
        self, 
        number_of_pcds: int = 2
        ):
        pcds = []
        skeleton = []
        for data in self.tf_dataset.take(number_of_pcds):
            points, skeletons = data
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.numpy())
            pcds.append(pcd)
            skeleton.append(skeletons.numpy())
        skeleton = pd.DataFrame(skeleton)

        from PyMoCapViewer import MoCapViewer
        viewer = MoCapViewer(grid_axis=None)
        viewer.add_point_cloud_animation(pcds)
        viewer.add_skeleton(skeleton)
        viewer.show_window()
      