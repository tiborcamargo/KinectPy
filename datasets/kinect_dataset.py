import os
import open3d as o3d
import pandas as pd
import tensorflow as tf
import numpy as np
#from preprocessing.utils import select_points_randomly
from typing import List, Tuple
np.set_printoptions(suppress=True)


def select_points_randomly(
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


class KinectDataset:
    def __init__(
        self,
        master_root_dirs: List[str],
        batch_size: int,
        joints: List[str],
        number_of_points: int,
        ):
        """
        Args:
            master_root_dirs: Path to the master's device root dir, eg: /path/to/master_1/
            batch_size: 
            joints: Skeleton joints defined by your RGB-D sensor, such as: ['PELVIS', 'FOOT', ...],
                    see all options in the 
            number_of_points: Number of points to be sub-sampled in a point cloud
        """
        # Dataset size will be increased when calling tf.data.Dataset.from_generator
        self.dataset_size = 0
        self.batch_size = batch_size
        self.joints = joints
        self.output_types = (tf.float32, tf.float32)
        self.number_of_joints = len(joints)
        self.number_of_points = number_of_points
        self.output_shapes = ((None, self.number_of_points, 3), (None, self.number_of_joints*3))
        
        for i in range(len(master_root_dirs)):
            filenames = os.listdir(os.path.join(master_root_dirs[i], 'filtered_and_registered_pointclouds'))
            self.dataset_size += len(filenames)

            tf_dataset = tf.data.Dataset.from_generator(
                    self._pointcloud_skeleton_tf_generator, 
                    args= [master_root_dirs[i], self.batch_size, self.number_of_points, self.joints],
                    output_types = self.output_types,
                    output_shapes = self.output_shapes
                    )

            # Concatenate all folders in one unique generator
            if i == 0:
                dataset_left = tf_dataset
            if i > 0:
                dataset_right =  tf_dataset
                dataset_left = dataset_left.concatenate(dataset_right)

        self.dataset = dataset_left


    def _pointcloud_skeleton_tf_generator(
        self,
        master_root_dir: str, 
        batch_size: int,
        number_of_points: int,
        joints: List[str]
        ):
        """
        Args:
            master_root_dir: Directory with master pointclouds, skeleton and other data
            batch_size: Size of batches when consuming the generator
            number_of_points: Downsampling a pointclod to use *number_of_points*
            number_of_joints: How many joints is used for the RGBD device
        """
        pcd_dir =  tf.io.gfile.join(master_root_dir.decode('utf-8'), 'filtered_and_registered_pointclouds')
        file_list =  tf.io.gfile.listdir(pcd_dir)

        skeleton_fp =  tf.io.gfile.join(master_root_dir.decode('utf-8'), 'skeleton', 'synced_positions_3d.csv')
        skeleton_df = pd.read_csv(open(skeleton_fp,'r'), index_col='timestamp')
        joints_columns = self._select_joints(joints, skeleton_df)
        skeleton_df = skeleton_df[joints_columns]
        
        np.random.shuffle(file_list)
    
        i = 0
        while True:
            if i*batch_size >= len(file_list):  
                i = 0
                np.random.shuffle(file_list)
            else:
                file_chunk = file_list[i*batch_size:(i+1)*batch_size] 
                pointcloud_points = []
                skeleton_positions = []
                for file in file_chunk:
                    # Retrieving timestamps to get skeletons positions
                    timestamp = int(file[:-4])
                    skeleton_positions.append(skeleton_df.loc[timestamp].values)
                    # Reading point cloud and sampling
                    pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, file))
                    pcd_points = select_points_randomly(pcd, number_of_points)
                    pointcloud_points.append(pcd_points)
                # Reshaping points 
                pointcloud_points = np.asarray(pointcloud_points)
                pointcloud_points = pointcloud_points.reshape(-1, number_of_points, 3) 

                skeleton_positions = np.asarray(skeleton_positions)
                skeleton_positions = skeleton_positions.reshape(-1, len(joints_columns))

                yield pointcloud_points, skeleton_positions
                i = i + 1


    def __call__(self):
        return self.dataset
    
    
    def _select_joints(
        self, 
        joints_list: List[str], 
        skeleton_dataframe: pd.DataFrame
        ):
        """ 
        Given a list of joints, select columns in the dataframe accordingly.
        This function is used because the name of the column might differ 
        from the joint name, 
            eg: joints_list = ['PELVIS']
            skelet_dataframe.columns
            >> ['PELVIS (x)', 'PELVIS (y)', 'PELVIS (z)']
        """
        joint_cols = []
        for col in skeleton_dataframe.columns:
            for joint in joints_list:
                if col.startswith(joint.decode('utf-8')):
                    joint_cols.append(col)
        return joint_cols


    def split_train_test_val(
        self,
        train_size: int = 0.7,
        val_size: int = 0.15,
        test_size: int = 0.15,
        shuffle: bool = True,
        ) -> List[tf.data.Dataset]:
        """
        Given a tensorflow dataset, return a split with train, test and validation datasets
    
        Args:
            train_size: Percentual of dataset to be used as training
            val_size: Percentual of dataset to be used as validation
            test_size: Percentual of dataset to be used as test
        
        Returns:
            (train_dataset, validation_dataset, test_dataset)
        """
        assert train_size + val_size + test_size == 1
    
        train_size = int(0.7 * self.dataset_size)
        val_size = int(0.15 * self.dataset_size)
        test_size = int(0.15 * self.dataset_size)
    
        train_dataset = self.dataset.take(train_size)
        test_dataset = self.dataset.skip(train_size)
        val_dataset = self.dataset.skip(val_size)
        test_dataset = self.dataset.take(test_size)

        return train_dataset, val_dataset, test_dataset


    def visualize_dataset(
        self, 
        number_of_batches: int = 2
        ):
        pcds = []
        skeleton = []
        for data in self.dataset.take(number_of_batches):
            points, skeletons = data
            for j in range(len(points)):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points.numpy()[j])
                pcds.append(pcd)
                skeleton.append(skeletons.numpy()[j])
        skeleton = pd.DataFrame(skeleton)

        from PyMoCapViewer import MoCapViewer
        viewer = MoCapViewer(grid_axis=None)
        viewer.add_point_cloud_animation(pcds)
        viewer.add_skeleton(skeleton)
        viewer.show_window()


#if __name__ == '__main__':
#
#    MASTER_ROOT_DIRS = [
#        'D:/azure_kinect/1E2DB6/02/master_1/',
#        'D:/azure_kinect/4AD6F3/01/master_1/', 'D:/azure_kinect/4AD6F3/02/master_1/',
#        'D:/azure_kinect/4B8AF1/01/master_1/', 'D:/azure_kinect/4B8AF1/02/master_1/',
#        'D:/azure_kinect/5E373E/01/master_1/', 'D:/azure_kinect/5E373E/02/master_1/',
#        ]
#
#    BATCH_SIZE = 16
#    JOINTS = ['FOOT_LEFT', 'FOOT_RIGHT', 'ANKLE_LEFT', 'ANKLE_RIGHT', 
#              'KNEE_LEFT', 'KNEE_RIGHT', 'HIP_LEFT', 'HIP_RIGHT', 'PELVIS']
#    NUMBER_OF_JOINTS = len(JOINTS)
#    NUMBER_OF_POINTS = 1024*8
#
#    kinect_dataset = KinectDataset(master_root_dirs=MASTER_ROOT_DIRS, batch_size=32)
#    dataset = kinect_dataset.dataset
#    train_ds, test_ds, val_ds = kinect_dataset.split_train_test_val(train_size=0.7, val_size=0.15, test_size=0.15)
#
#    for train_data in train_ds.take(1):
#        points, skeleton = train_data
#        print(f'Points: {points.shape}')
#        print(f'Skeleton: {skeleton.shape}')
#        break
#
#    kinect_dataset.visualize_dataset(2)
