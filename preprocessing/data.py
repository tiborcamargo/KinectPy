import os
import cv2
import logging
import numpy as np
import pandas as pd
import open3d as o3d
from preprocessing.filtering import Filtering
from utils.processing import sort_filenames_by_timestamp
from preprocessing.registration import execute_global_registration, execute_local_registration
from typing import List


class DataProcessor:
    def __init__(
        self, 
        output_dirs: List[str]
        ):
        """
        Finds a registration matrix between master and sub devices, 
        filter all color images, register all filtered point clouds 
        and then save to disk

        output path: /path/to/master_1/filtered_and_registered_pointclouds
        """
        self.device_filenames_df = self._create_device_filenames_df(output_dirs)
        self.number_of_devices = len(self.device_filenames_df.columns)
        self.registration_transformations = []
        self._find_registration_transforms()
        logging.info('Starting to filter background and save filtered point clouds')
        self.segmentation = Filtering('F:/frozen_inference_graph.pb', 
                                      'F:/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')
        for file_idx in range(len(self.device_filenames_df)):
            if (file_idx+1)%50 == 0:
                print(f'{file_idx} point clouds have been saved')
            filtered_pcds = self._filter_pointclouds_and_save(file_idx)
            registered_pcd_points = []
            registered_pcd_colors = []
            registered_pcd = o3d.geometry.PointCloud()
            for device_idx in range(self.number_of_devices):
                # Registering master and sub devices
                if device_idx == 0:
                    registered_pcd = filtered_pcds[device_idx]
                else:
                    registered_pcd.transform(self.registration_transformations[device_idx-1])
                # Retrieving their point
                registered_pcd_points.append(np.asarray(registered_pcd.points))
                registered_pcd_colors.append(np.asarray(registered_pcd.colors))
            # Transforming to o3d vector
            registered_pcd_points = o3d.utility.Vector3dVector(np.vstack(registered_pcd_points))
            registered_pcd_colors = o3d.utility.Vector3dVector(np.vstack(registered_pcd_colors))
            registered_and_filtered_pcd_fp = os.path.join(
                self.device_filenames_df.columns[0], 
                'filtered_and_registered_pointclouds', 
                self.device_filenames_df.iloc[file_idx][0]
            )
            o3d.io.write_point_cloud(registered_and_filtered_pcd_fp + '.pcd', registered_pcd)


    def _create_device_filenames_df(self, output_dirs):
        """ 
        Creates a dataframe with all correspondences between frames of each device
        """
        device_filename_mapper = {}
        color_suffix = '_rgb.png'
        for output_dir in output_dirs:
            color_dir = os.path.join(output_dir, 'color')
            filenames = sort_filenames_by_timestamp([x.split(color_suffix)[0] for x in os.listdir(color_dir)])        
            device_filename_mapper[output_dir] = filenames
        filenames_df = pd.DataFrame(device_filename_mapper)
        return filenames_df
    

    def _filter_pointclouds_and_save(self, file_idx):
        """ 
        Returns a list containing all filtered point clouds,
        following the structure:
        [filtered_master_pcd, filtered_sub_1_pcd, filtered_sub_2_pcd, ...]
        """
        filtered_pointclouds = []
        # Create master device point cloud
        master_root_dir = self.device_filenames_df.columns[0]
        # For each timestamp, filter image, save to pointcloud and then save
        master_color_fp = os.path.join(master_root_dir, 'color', self.device_filenames_df.iloc[file_idx][0])
        master_depth_fp = os.path.join(master_root_dir, 'depths', self.device_filenames_df.iloc[file_idx][0])
        master_color = self._load_color(master_color_fp)
        master_depth = self._load_depth(master_depth_fp)

        # Apply filtering
        master_filtered_img = self.segmentation.apply_segmentation(master_color)
        master_pcd = self._transform_filtered_image_to_pointcloud(master_filtered_img, master_depth)
        filtered_pointclouds.append(master_pcd)
        
        # For each sub device
        for device_idx in range(1, self.number_of_devices):
            # Create sub device point cloud
            sub_root_dir = self.device_filenames_df.columns[device_idx]
            sub_color_fp = os.path.join(sub_root_dir, 'color', self.device_filenames_df.iloc[file_idx][device_idx])
            sub_depth_fp = os.path.join(sub_root_dir, 'depths', self.device_filenames_df.iloc[file_idx][device_idx])
            sub_color = self._load_color(sub_color_fp)
            sub_depth = self._load_depth(sub_depth_fp)                                                                            

            # Apply filtering
            sub_filtered_img = self.segmentation.apply_segmentation(sub_color)
            sub_pcd = self._transform_filtered_image_to_pointcloud(sub_filtered_img, sub_depth)
            filtered_pointclouds.append(sub_pcd)
            
        return filtered_pointclouds
        

    def _find_registration_transforms(self):
        """
        Apply global and then local registration algorithm, 
        saving the transformations to disk
        """

        # Finding registratoin matrix based on the 0-th image 
        file_idx = 0
        # Create master device point cloud
        master_root_dir = self.device_filenames_df.columns[0]
        master_color_fp = os.path.join(master_root_dir, 'color', self.device_filenames_df.iloc[file_idx][0])
        master_depth_fp = os.path.join(master_root_dir, 'depths', self.device_filenames_df.iloc[file_idx][0])

        master_color = self._load_color(master_color_fp)
        master_depth = self._load_depth(master_depth_fp)
        master_pcd = self._rgbd_to_pointcloud(master_color, master_depth)
        
        
        for device_idx in range(1, self.number_of_devices):
            
            # Create sub device point cloud
            sub_root_dir = self.device_filenames_df.columns[device_idx]
            sub_color_fp = os.path.join(sub_root_dir, 'color', self.device_filenames_df.iloc[file_idx][device_idx])
            sub_depth_fp = os.path.join(sub_root_dir, 'depths', self.device_filenames_df.iloc[file_idx][device_idx])
            
            sub_color = self._load_color(sub_color_fp)
            sub_depth = self._load_depth(sub_depth_fp)                                                                            
            sub_pcd = self._rgbd_to_pointcloud(sub_color, sub_depth)
               
            # Apply registration and save the transformation matrix
            initial_transformation = execute_global_registration(master_pcd, sub_pcd)
            local_transformation = execute_local_registration(master_pcd, sub_pcd, initial_transformation)
            transformation_dst = os.path.join(self.device_filenames_df.columns[0],
                                             f'transformation_master_sub_{device_idx}.npy')
            np.save(transformation_dst, local_transformation)
            self.registration_transformations.append(local_transformation)
            
        
        
    def _transform_filtered_image_to_pointcloud(self, filtered_img, depth_img):
        reshaped_color = filtered_img.reshape(-1, 3)

        # Select non-black pixels and within a distance of +- 200 cm from the median depth
        valid_pixels = (reshaped_color[:, 0] != 0) & (reshaped_color[:, 1] != 0) & (reshaped_color[:, 2] != 0)
        valid_depths = (depth_img[:, 2] <= np.median(depth_img[:, 2]) + 750) | \
                       (depth_img[:, 2] <= np.median(depth_img[:, 2]) - 750)

        # Filter out black pixels and invalid depths
        reshaped_color = reshaped_color[(valid_pixels) & (valid_depths)]
        depth_img = depth_img[(valid_pixels) & (valid_depths)]
        
        pcd = self._rgbd_to_pointcloud(reshaped_color, depth_img)
        return pcd
        
        
    def _load_color(self, color_fp):
        color_suffix = '_rgb.png'
        if not color_fp.endswith(color_suffix):
            color_fp += color_suffix
        color = cv2.imread(color_fp)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        return color
    
    
    def _load_depth(self, depth_fp):
        depth_suffix = '_depth.dat'
        if not depth_fp.endswith(depth_suffix):
            depth_fp += depth_suffix
        depth = np.fromfile(depth_fp, dtype=np.int16).reshape(-1, 3)
        return depth
    
    
    def _rgbd_to_pointcloud(self, color_img, depth_img):
        """
        Transforms RGB+D to point cloud, also removing color values
        of rgb = [0, 0, 0], since many of them are invalid
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(depth_img)
        pcd.colors = o3d.utility.Vector3dVector(color_img.reshape(-1, 3) / 255)
        
        # removing black color
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        non_black_idx = (points[:, 0] != 0) & (points[:, 1] != 0) & (points[:, 2] != 0)
        valid_points = points[non_black_idx]
        valid_colors = colors[non_black_idx]

        pcd.points = o3d.utility.Vector3dVector(valid_points)
        pcd.colors = o3d.utility.Vector3dVector(valid_colors)
        
        return pcd
