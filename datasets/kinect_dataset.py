import os
import glob
import open3d as o3d
import pandas as pd
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Literal
np.set_printoptions(suppress=True)


class KinectDataset:
    def __init__(
        self,
        master_root_dirs: List[str],
        joints: List[str],
        number_of_points: int,
        flag: Literal['train', 'val', 'test']
        ):
        """
        Args:
            master_root_dirs: Path to the master's device root dir, eg: /path/to/master_1/
            batch_size: 
            joints: Skeleton joints defined by your RGB-D sensor, such as: ['PELVIS', 'FOOT', ...],
                    see all options in the 
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

        # Verify if flag corresponds to the directory structure being passed
        pcd_dirs = []
        self.pointcloud_files = []
        for master_root_dir in master_root_dirs:

            # asserting if train/val/test is passed correctly
            if flag not in os.path.abspath(master_root_dir).split(os.path.sep):
                 raise Exception('Directory structure does not match the dataset flag')

            # Point cloud directories and its corresponding skeleton CSV files
            pcd_dir_regex = os.path.abspath(os.path.join(master_root_dir, 'filtered_and_registered_pointclouds'))
            pcd_dirs.append(glob.glob(pcd_dir_regex))
            self.pointcloud_files.append(glob.glob(os.path.join(pcd_dir_regex, '*.pcd')))

        # transforming lists to numpy array and shuffling pcd filepaths
        pcd_dirs = np.concatenate(pcd_dirs)
        self.pointcloud_files = np.concatenate(self.pointcloud_files)
        self.dataset_size = len(self.pointcloud_files)
        np.random.shuffle(self.pointcloud_files)

        # Mapping filename to a skeleton CSV
        self.correspondent_skeleton_csv = {}
        for pcd_dir in pcd_dirs:
            skeleton_fp = pcd_dir.replace('filtered_and_registered_pointclouds', os.path.join('skeleton', 'synced_positions_3d.csv'))
            self.correspondent_skeleton_csv[pcd_dir] = pd.read_csv(skeleton_fp, index_col='timestamp')

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
            print(file, flush=True)

            # Finding the proper skeleton dataframe and the proper timestamp 
            timestamp = int(file.split(os.path.sep)[-1][:-4])
            correspondent_skeleton_key = tf.io.gfile.join(
                os.path.sep.join(file.split(os.path.sep)[:-2]), 
                'filtered_and_registered_pointclouds'
            )

            # Retrieving skeletons positions and joints
            skeleton_df = self.correspondent_skeleton_csv[correspondent_skeleton_key]
            skeleton_positions = skeleton_df[self.joints_columns].loc[timestamp].values

            ## Reading point cloud and sampling
            pcd = o3d.io.read_point_cloud(file)
            pcd_points = self._select_points_randomly(pcd, number_of_points)
                
            yield pcd_points, skeleton_positions
            i = i + 1
                
                
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
      