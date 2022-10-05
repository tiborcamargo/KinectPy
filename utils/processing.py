import os
import copy
import shutil
import numpy as np
import pandas as pd
import open3d as o3d
from typing import List, Tuple, Union

from zmq import device

    
def sort_filenames_by_timestamp(
    list_of_files: str
    ) -> List[str]:
    """ 
    Given a list of files with numbers, such as 1232.png or 0023123.jpg, 
    sort them and return the sorted ordering 
    """
    list_of_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    return np.array(list_of_files)


def find_delay_master_sub(
    timestamps_dataframe: pd.DataFrame
    ) -> List[int]:
    """
    Given a pandas dataframe containing the sorted timestamps for all point clouds 
    in each device, find the delay between master and sub devices, for examples:
        ```
        master_1, sub_1, sub_2
          001      005    001
          004      006    004           

        >> [1, 0] 
        Because sub_1 (003) is closest to the master_1 (002)
        and sub_2 (001) is closest to the master_1 (001)
        ```
    """
    idx_delays = []
    for device in timestamps_dataframe.columns[1:]:
        first_timestamp_for_sub = int(timestamps_dataframe[[device]].iloc[0])
        min_delay = np.inf
        idx_delay = -1
        for i, row in enumerate(timestamps_dataframe[['master_1']].itertuples()):
            master_timestamp = row[1]
            timestamp_delay = abs(first_timestamp_for_sub - master_timestamp)
            if timestamp_delay < min_delay:
                min_delay = timestamp_delay
                idx_delay = i
        idx_delays.append(idx_delay)
    return idx_delays


def synchronize_filenames(
    output_dirs: List[str]
    ) -> pd.DataFrame:
    """
    Given a collection of directories with the extracted pointclouds from MKV files,
    returns a DataFrame where the filenames (or timestamps) are aligned side-by-side.
    """
    if type(output_dirs) != type([]):
        output_dirs = [output_dirs]

    color_suffix = '_rgb.png'
    depth_suffix = '_depth.dat'
    
    timestamp_device_map = {}
    dataframes = []

    for device_dir in output_dirs:
        # transform /path/to/master_1 (sub_1) in absolute path
        device_dir = os.path.abspath(device_dir)
        
        # extract which device
        device_name = os.path.basename(device_dir)

        # point to depth and rgb folders
        color_dir = os.path.join(device_dir, 'color')
        depth_dir = os.path.join(device_dir, 'depths')
        skel_dir = os.path.join(device_dir, 'skeleton')

        fns = [x.split(color_suffix)[0] for x in os.listdir(color_dir) if x.endswith('_rgb.png')]
        sorted_filename_prefixes = sort_filenames_by_timestamp(fns)

        timestamp_device_map[device_name] = sorted_filename_prefixes
        df = pd.DataFrame({'timestamp':sorted_filename_prefixes})

    df = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in timestamp_device_map.items()])).astype(float)
    # Removing rows where all values are equal to 0 (due to a problem with extraction)
    df = df.loc[(df != 0).all(axis=1)] 

    # Synchronizing the dataframe
    idx_delays = find_delay_master_sub(df)
    for i, idx_delay in enumerate(idx_delays):
        sub_device = i+1
        df[f'sub_{sub_device}'] = df[f'sub_{sub_device}'].shift(idx_delays[i])
        
    return df


def remove_files_not_synced(
    output_dirs
    ) -> None:
    """ 
    Given the output_dirs: 
        /path/to/master_1, /path/to/sub_1, /path/to/sub_1, ...
        
    Remove files that are not synced.
    
    Example of synced_filenames_df:
        master_1 sub_1
           001    nan
           002    nan
           003    003 
           
    We then delete the color and depth images from 001 and 002 on master_1, and 
    also drop their indices on the corresponding skeleton files

    After this stage, all files can be synced by using a sorted list of indices
    """
    synced_filenames_df = synchronize_filenames(output_dirs)
    output_dirs = [os.path.abspath(output_dir) for output_dir in output_dirs]
    color_suffix = '_rgb.png'
    depth_suffix = '_depth.dat'
    
    number_of_devices = len(output_dirs)
    for row in synced_filenames_df.iterrows():
        index = row[0]  # contains index of the dataframe
        values = row[1].values # contains the row values

        # If any value in a row is "nan", then the folders are not synced
        # and we need to remove the corresponding files
        if np.isnan(values).any():
            for dir_idx, timestamp in enumerate(values):
                color_dir = os.path.join(output_dirs[dir_idx], 'color')
                depths_dir = os.path.join(output_dirs[dir_idx], 'depths')

                # We delete only the not-nan values
                if not np.isnan(timestamp):
                    remove_color_fp = os.path.join(color_dir, str(int(timestamp)) + color_suffix)
                    remove_depth_fp = os.path.join(depths_dir, str(int(timestamp)) + depth_suffix)
                    os.remove(remove_color_fp)
                    os.remove(remove_depth_fp)


def align_master_and_sub_frames(
    m_files: List[str], 
    s_files: List[str]
    ) -> Tuple[List[str], List[str]]:
    
    # First sort the master and sub files
    sorted_master = sort_filenames_by_timestamp(m_files)
    sorted_sub = sort_filenames_by_timestamp(s_files)
    
    return sorted_master, sorted_sub


def find_correspondent_frames_between_folders(
    files_dirs: List[str]
    ) -> dict:
    """
    Finds which frames (or files) corresponds to the same scene.
    This is needed because different sensors may capture rgb/depth images with custom timestamps,
    so a synchronization is needed.
    
    obs: files named "0.pcd" are ignored because of a bug from offline_processor.exe ***
    
    Args:
        files_dirs: Directories with the files you want to align, starting 
                    with the master directorory, sub 1, sub 2, and so on
                    Example: ['path/to/master_1/images', 'path/to/sub_1/images'] 
    """
    correspondence_master_and_sub_files = {}
    # get the master files
    master_files = np.array([fn for fn in os.listdir(files_dirs[0]) \
                             if fn != '0.pcd'][:])
    for device_idx in range(1, len(files_dirs)):
        sub_files = np.array([fn for fn in os.listdir(files_dirs[device_idx]) \
                              if fn != '0.pcd'][:])
        aligned_master, aligned_sub = align_master_and_sub_frames(master_files, sub_files)
        correspondence_master_and_sub_files[f'master_to_sub_{device_idx}'] = (aligned_master, aligned_sub)
    return correspondence_master_and_sub_files


def scale_point_cloud(
    pcd: o3d.geometry.PointCloud, 
    xs: float=0.001, 
    ys: float=0.001, 
    zs: float=0.001
    ) -> o3d.geometry.PointCloud:
    scaled_pcd = copy.deepcopy(pcd)
    pcd_new_points = np.asarray(scaled_pcd.points)
    pcd_new_points[:, 0] = xs*np.asarray(scaled_pcd.points)[:, 0]
    pcd_new_points[:, 1] = ys*np.asarray(scaled_pcd.points)[:, 1]
    pcd_new_points[:, 2] = zs*np.asarray(scaled_pcd.points)[:, 2]
    scaled_pcd.points = o3d.utility.Vector3dVector(pcd_new_points)
    return scaled_pcd

    
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([source_temp, target_temp, cf])

    
def labels_from_csv(
    skeleton_filepath: str
    ) -> pd.DataFrame:
    """ 
    Takes the 'positions_3d.csv' skeleton file and extract as dataframe 
    """ 
    skeleton = pd.read_csv(skeleton_filepath, sep=';')
    skeleton = skeleton.set_index('timestamp')
    skeleton = skeleton[[col for col in skeleton.columns \
                         if not col.endswith('(w)') and col != 'body_idx' \
                         and not col.endswith('(c)')]]
    return skeleton


def sync_skeleton_and_pointcloud(
    root_dirs: Union[str, List[str]], 
    get_confidence_intervals: bool = False,
    save_csv: bool=False
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Given a root dir, such as '/path/to/master_1', aligns timestamps for
    the skeleton estimation file to the point clouds  
    """
    if type(root_dirs) == str:
        root_dirs = [root_dirs]

    for root_dir in root_dirs:
        color_dir = os.path.join(root_dir, 'color')
        color_filenames = os.listdir(color_dir)
        sorted_color_filenames = sort_filenames_by_timestamp(color_filenames)
    
        timestamps = [int(fn.split('_rgb.png')[0]) for fn in sorted_color_filenames]
        timestamps_df = pd.DataFrame({'timestamp':timestamps})

        skeleton_fp = os.path.join(root_dir, 'skeleton', 'positions_3d.csv')
        skeleton_df = pd.read_csv(skeleton_fp, sep=';')
        
        if get_confidence_intervals:
            confidence_interval_df = skeleton_df[[col for col in skeleton_df.columns \
                if col.endswith('(c)') and col != 'body_idx']]

        skeleton_df = skeleton_df[[col for col in skeleton_df.columns \
                                   if not col.endswith('(c)') and col != 'body_idx']]


        synced_pcd_skeleton = pd.merge_asof(timestamps_df, skeleton_df, on='timestamp')
        synced_pcd_skeleton = synced_pcd_skeleton.dropna()
        synced_pcd_skeleton = synced_pcd_skeleton.set_index('timestamp')
    
        if save_csv:
            saved_fp = os.path.join(root_dir, 'skeleton', 'synced_positions_3d.csv')
            synced_pcd_skeleton.to_csv(saved_fp, index=True)
        
    if get_confidence_intervals:
        return synced_pcd_skeleton, confidence_interval_df
    else:
        return synced_pcd_skeleton
        

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


# TODO: Remove useless dirs for ALL devices
def remove_useless_dirs(root_dir):
    """ 
    Remove the color, depths, pointclouds and filtered_pointclouds directories,
    since they are not used after filtering/registration/alignment
    """
    for device in ['master_1', 'sub_1']:
        dirs_to_remove = [
            os.path.join(root_dir, device, 'color'),
            os.path.join(root_dir, device, 'depths'),
            os.path.join(root_dir, device, 'pointclouds'),
            os.path.join(root_dir, device, 'filtered_pointclouds')
        ]

        # Try to remove, if it does not exist just ignore
        for folder in dirs_to_remove:
            print(folder)
            try:
                shutil.rmtree(folder)
            except:
                pass
    return


def statistical_outlier_removal(pcd, nb_neighbors: int = 200, std_ratio: float = 3.0):
    """ 
    Applies a statistical outlier removal for the Point Cloud `pcd`.
    This method is specially useful to be applied before using any 
    procedure that involves Oriented Bounding Boxes
    """
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    filtered_cloud, outlier_idx = voxel_down_pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    return filtered_cloud


def normalize_pointcloud(
    pcd: o3d.geometry.PointCloud, 
    min_range: float = -1.0, 
    max_range: float = 1.0
    ) -> o3d.geometry.PointCloud:
    """ 
    Given a point set P = {(x1,y1,z1), ..., (xN,yN,zN)}, 
    ensures that all points are linearly set to the range [min_range, max_range]
    """
    arr = np.asarray(pcd.points)
    scaled_unit = (max_range - min_range) / (np.max(arr) - np.min(arr))
    scaled_points = arr*scaled_unit - np.min(arr)*scaled_unit + min_range
    pcd.points = o3d.utility.Vector3dVector(scaled_points)
    return pcd


def obb_normalization(
    points: np.ndarray, 
    joints: np.ndarray, 
    number_of_joints: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """ 
    Retrieves the center and the orientation of an Oriented Bounding Box 
    to normalize point clouds
    """
    # Create temp pcd to retrieve OBB translation + rotation
    tmp_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
    
    # Retrieve rotation and translation
    obb = tmp_pcd.get_oriented_bounding_box()
    obb_rot = obb.R
    obb_trans = obb.get_center()

    # Normalize points
    obb_normalized_points = (points - obb_trans)@obb_rot
    
    # Normalize joints
    joints_3d = joints.values.reshape((number_of_joints, 3))
    obb_normalized_joints = (joints_3d - obb_trans)@obb_rot
    obb_normalized_joints = obb_normalized_joints.reshape((number_of_joints*3))
    
    return obb_normalized_points, obb_normalized_joints


def transform_joints(
    skeleton_df: pd.DataFrame, 
    transformation: np.ndarray
    ) -> pd.DataFrame:
    '''
    Take a 4x4 (rigid) transformation and apply to `skeleton_df`
    '''
    number_of_joints = len(skeleton_df.columns)//3
    rotation = transformation[:3, :3]
    translation = transformation[:3, 3]

    # convert dataframe to a (n frames, k joints, 3) shaped data and then 
    transformed_skeleton = skeleton_df.values.reshape(
        (skeleton_df.shape[0], number_of_joints , 3)
        )
    transformed_skeleton = transformed_skeleton@np.linalg.inv(rotation) + translation

    # transforming it back to the (n frames, k*3) shaped data
    # columns: [joint_1x, joint_1y, joint_1z, joint_2x, ..., joint_kx, joint_ky, joint_kz]
    transformed_skeleton = transformed_skeleton.reshape(
        (skeleton_df.shape[0], number_of_joints*3)
        )
    transformed_skeleton = pd.DataFrame(columns=skeleton_df.columns, 
                                        data=transformed_skeleton,
                                        index=skeleton_df.index)

    return transformed_skeleton


def synchronize_joints(
    root_dirs: List[str], 
    transformations: Union[None, List[np.ndarray]], 
    joint_names: List[str],
    get_confidence_intervals: bool
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    ''' 
    Given a list of root_dirs (master_1/sub_1/sub_2/...), 
    load all skeleton dataframes and apply their correspondent
    registration transformations.
    
    Transformations are expected to be from master to sub, such as:
        'transform_master_sub_1', 'transform_master_sub_2'

    get_confidence_intervals will make this function return 
    an additional point cloud with confidence intervals from
    the body tracking SDK, with
    1 =
    '''
    joints_columns = np.concatenate([[name + ' (x)', name + ' (y)', name + ' (z)']  for name in joint_names])
    confidence_columns = np.array([name + ' (c)' for name in joint_names])

    synced_filenames = synchronize_filenames(root_dirs).dropna().astype(int)
    skeleton_dfs = []
    confidence_intervals = []
    for i in range(len(root_dirs)):
        device = synced_filenames.columns[i]
        if get_confidence_intervals:
            synced_joints_df, confidence_interval = sync_skeleton_and_pointcloud(root_dirs[i], get_confidence_intervals=True)
            confidence_interval = confidence_interval[confidence_columns]
            confidence_intervals.append(confidence_interval)
        else:
            synced_joints_df = sync_skeleton_and_pointcloud(root_dirs[i], get_confidence_intervals=False)
        skeleton_df = synced_joints_df[joints_columns].loc[synced_filenames[device]]
        if i > 0 and transformations[i-1] is not None:
            skeleton_df = transform_joints(skeleton_df, transformations[i-1])
        skeleton_dfs.append(skeleton_df)
        
    if get_confidence_intervals:
        return skeleton_dfs, confidence_intervals
    else:
        return skeleton_dfs
        