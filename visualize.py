# Usage example: python visualize.py -d "F:/4AD6F3" -s
import os
import open3d as o3d
import pandas as pd
from argparse import ArgumentParser
from PyMoCapViewer import MoCapViewer
from utils.processing import sort_filenames_by_timestamp


parser = ArgumentParser()
parser.add_argument("-d", "--dir", help="root directory", type=str)
parser.add_argument("-s", "--skeleton", help="display skeleton", action='store_true')
args = parser.parse_args()


if __name__ == '__main__':
    
    root_dir = args.dir
    pcd_dir = os.path.join(root_dir, 'master_1/filtered_and_registered_pointclouds')
    skeleton_fp = os.path.join(root_dir, 'master_1/skeleton/registered_positions_3d.csv')

    pcd_filenames = sort_filenames_by_timestamp(os.listdir(pcd_dir))
    pcds = [o3d.io.read_point_cloud(os.path.join(pcd_dir, fn)) for fn in pcd_filenames]

    viewer = MoCapViewer(grid_axis=None)
    viewer.add_point_cloud_animation(pcds)

    if args.skeleton:
        df = pd.read_csv(skeleton_fp, index_col='timestamp')
        viewer.add_skeleton(df)

    viewer.show_window()
