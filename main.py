import os
import time
from preprocessing.extractor import MKVFilesProcessing
from preprocessing.data import DataProcessor

# Path to the Azure Kinect offline processor
OFFLINE_PROCESSOR_PATH = os.path.join(
        'C:/', 'Users', 'Tibor', 'source', 'repos', 'Azure-Kinect-Samples', 'body-tracking-samples', 
        'Azure-Kinect-Extractor', 'build', 'bin', 'Debug', 'offline_processor.exe'
        )

NUMBER_OF_JOINTS = 32 

if __name__ == '__main__':

    MKV_EXPERIMENTS_DIR = [
        'F:/1E2DB6/',
        'F:/4AD6F3',
        'F:/4B8AF1',
        'F:/5E373E',
    ]

    start = time.time()
    for experiment_dir in MKV_EXPERIMENTS_DIR:

        mkv_input_files = [os.path.join(experiment_dir, 'master_1.mkv'), os.path.join(experiment_dir, 'sub_1.mkv')]
        mkv_output_dirs = [x.replace('.mkv', '') for x in mkv_input_files]

        extractor = MKVFilesProcessing(mkv_input_files, 
                                       mkv_output_dirs, 
                                       OFFLINE_PROCESSOR_PATH,
                                       NUMBER_OF_JOINTS) 
    
        # Extract pointclouds, color,depths, skeleton from the MKV file
        extractor.extract(pointcloud=False, skeleton=True)

        #data_processor = DataProcessor(mkv_output_dirs)

        # Aligning skeletons after registration has been done
        extractor.align_skeletons() 
        print(f'It took {round(time.time() - start, 2)} seconds to process {len(mkv_input_files)} MKV files.')
