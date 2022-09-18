import argparse
from configs.config import get_default_config

# Give path to the yaml configuration file using argument parser
parser = argparse.ArgumentParser(description=__doc__, add_help=True)
parser.add_argument('--config', help='configuration file *.yaml', required=True)

# Loading configuration and defining constants
CONFIGS = get_default_config(parser)
EPOCHS = CONFIGS.TRAIN.EPOCHS
BATCH_SIZE = CONFIGS.TRAIN.BATCH_SIZE
JOINTS = CONFIGS.DATA.JOINTS
NUMBER_OF_POINTS = CONFIGS.DATA.SAMPLING_POINTS
NORMALIZATION = CONFIGS.DATA.NORMALIZATION
LEARNING_RATE = CONFIGS.TRAIN.LEARNING_RATE
THRESHOLDS = CONFIGS.EVAL.THRESHOLDS
LOSS = CONFIGS.TRAIN.LOSS
METRICS = CONFIGS.EVAL.METRICS
PROJECT = CONFIGS.PROJECT.NAME
NAME = CONFIGS.PROJECT.EXPERIMENT
CKPT_DIR = CONFIGS.DIRS.CHECKPOINT_DIR
LOGS_DIR = CONFIGS.DIRS.LOGS_DIR

# Import after argument parser 
import os
import glob
import logging
import numpy as np
import logging.config
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from models.pointnet import create_pointnet
from datasets.kinect_dataset import KinectDataset
from metrics.metric import percentual_correct_keypoints
from options.normalization import normalization_options

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.set_printoptions(suppress=True)
tf.random.set_seed(1234)
logging.config.dictConfig({
    'version': 1,
    # Other configs ...
    'disable_existing_loggers': True
})

# Data
TEST_ROOT_DIRS = glob.glob('D:/azure_kinect/test/*')

if __name__ == '__main__':

    if NORMALIZATION == 'obb_normalization':
        raise NotImplementedError('Needs to apply inverse mapping to evaluate')

    # Create model, compile and load
    pck_start = 0
    pck_end = 200
    metrics = [percentual_correct_keypoints(t) for t in range(pck_start, pck_end + 10, 10)]
    model = create_pointnet(NUMBER_OF_POINTS, len(JOINTS))

    checkpoint_dir = os.path.join(CKPT_DIR, PROJECT, NAME)
    logging.info(f'Loading ckpt from {checkpoint_dir}')

    model.load_weights(checkpoint_dir)
    model.compile(
        loss=LOSS,
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=metrics
    )

    # Import dataset
    test_dataset = KinectDataset(
        subjects_dirs=TEST_ROOT_DIRS, 
        number_of_points=NUMBER_OF_POINTS,
        joints=JOINTS,
        flag='test'
    )
    
    test_ds = test_dataset().batch(BATCH_SIZE).prefetch(1)

    if NORMALIZATION != '':
        test_ds = test_ds.map(normalization_options[NORMALIZATION])    

    evaluation = model.evaluate(test_ds, steps=test_dataset.dataset_size//BATCH_SIZE, verbose=1)
    test_loss = evaluation[0]
    test_pck = evaluation[1:]

    # Save and print evaluation
    result_message = f'Mean loss = {str(test_loss)}\n'
    for metric, pck in zip(metrics, test_pck):
        result_message += f"PCK@{''.join(filter(str.isdigit, metric.__name__))}, Mean PCK = {str(pck)}" + '\n'

    evaluation_fp = os.path.join(
        checkpoint_dir,
        "evaluation.txt"
        )

    with open(evaluation_fp, 'w') as file:
        file.write(result_message)
    
    plt.plot([t for t in range(pck_start, pck_end + 10, 10)], evaluation[1:])
    plt.title(NAME)
    plt.xlabel('threshold (milimiters)')
    plt.ylabel('pck')
    plt.ylim([0, 1])
    plt.xlim([pck_start, pck_end])
    plt.grid()
    plt.savefig(evaluation_fp.replace('.txt', '.png'))
    plt.close()

    print('Data saved at: ' + evaluation_fp, flush=True)
    logging.info(result_message)
