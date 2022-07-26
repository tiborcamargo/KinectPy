import os
import copy
import time
import pandas as pd
import numpy as np
import open3d as o3d
from preprocessing.extractor import MKVFilesProcessing
from preprocessing.data import DataProcessor

# Path to the Azure Kinect offline processor
OFFLINE_PROCESSOR_PATH = os.path.join(
        'C:/', 'Users', 'Tibor', 'source', 'repos', 'Azure-Kinect-Samples', 'body-tracking-samples', 
        'Azure-Kinect-Extractor', 'build', 'bin', 'Debug', 'offline_processor.exe'
        )

MKV_INPUT_FILES = [
    'F:/Kinect_data_19052022/debug/cmj_new/master_1.mkv',
    'F:/Kinect_data_19052022/debug/cmj_new/sub_1.mkv',
    #'F:/Kinect_data_19052022/debug/cmj_new/sub_2.mkv',
]

MKV_OUTPUT_DIRS = [x.replace('.mkv', '') for x in MKV_INPUT_FILES]

NUMBER_OF_JOINTS = 32

if __name__ == '__main__':
    start = time.time()
    extractor = MKVFilesProcessing(MKV_INPUT_FILES, 
                                   MKV_OUTPUT_DIRS, 
                                   OFFLINE_PROCESSOR_PATH,
                                   NUMBER_OF_JOINTS) 
    
    # Extract pointclouds, color, depths, skeleton from the MKV file
    extractor.extract(pointcloud=True, skeleton=True)

    data_processor = DataProcessor(MKV_OUTPUT_DIRS)

    # Aligning skeletons after registration has been done
    #extractor.align_skeletons()
    print(f'It took {round(time.time() - start, 2)} seconds to process {len(MKV_INPUT_FILES)} MKV files.')
