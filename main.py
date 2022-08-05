import os
import time
import logging
from preprocessing.extractor import MKVFilesProcessing
from preprocessing.data import DataProcessor
from utils.processing import remove_useless_dirs

try:
    os.mkdir('logs')
except:
    pass

logging.basicConfig(filename='logs/main.log', 
                    level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    force=True)

logger = logging.getLogger(__name__)

# Path to the Azure Kinect offline processor
OFFLINE_PROCESSOR_PATH = os.path.join(
        'C:/', 'Users', 'Tibor', 'source', 'repos', 'Azure-Kinect-Samples', 'body-tracking-samples', 
        'Azure-Kinect-Extractor', 'build', 'bin', 'Debug', 'offline_processor.exe'
        )

NUMBER_OF_JOINTS = 32 

if __name__ == '__main__':

    MKV_EXPERIMENTS_DIR = [
        #'F:/F205FE/',
        #'F:/1E2DB6/',
        'F:/4AD6F3',
        'F:/4B8AF1',
        'F:/5E373E',
        'F:/F205FE',
        'F:/76ABFD',
        'F:/37A7AA',
        'F:/20E29D',
        'F:/9AE368',
        'F:/339F94',
        'F:/471EF1',
        'F:/857F1E',
        'F:/927394',
        'F:/AEBA3A',
        'F:/AFCD31',
        'F:/C47EFC',
        'F:/CCB8AD',
        'F:/EEFE6D',
    ]

    for experiment_dir in MKV_EXPERIMENTS_DIR:

        try:
            logging.info(f'Starting experiment: {experiment_dir}')
            mkv_input_files = [os.path.join(experiment_dir, 'master_1.mkv'), os.path.join(experiment_dir, 'sub_1.mkv')]
            mkv_output_dirs = [x.replace('.mkv', '') for x in mkv_input_files]
               
            extractor = MKVFilesProcessing(mkv_input_files, 
                                           mkv_output_dirs, 
                                           OFFLINE_PROCESSOR_PATH,
                                           NUMBER_OF_JOINTS) 
            
            # Extract pointclouds, color,depths, skeleton from the MKV file
            extractor.extract(pointcloud=True, skeleton=True)

            # Filter and register point clouds
            data_processor = DataProcessor(mkv_output_dirs)
            
            # Aligning skeletons after registration has been done
            extractor.align_skeletons() 

            # remove colors, depths
            remove_useless_dirs(experiment_dir)

        except Exception as err:
            logger.error(err)
