import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from models.pointnet import create_pointnet
from datasets.kinect_dataset import KinectDataset
from metrics.metric import percentual_correct_keypoints
from configs.argparser import parse_args 
np.set_printoptions(suppress=True)
tf.random.set_seed(1234)


TEST_ROOT_DIRS = [
    'D:/azure_kinect/76ABFD/03/master_1', 
    'D:/azure_kinect/76ABFD/04/master_1', 
    'D:/azure_kinect/CCB8AD/01/master_1', 
]


if __name__ == '__main__':

    # Read arguments
    configs = parse_args(print_config=True)

    if configs['datasets']:
        TEST_ROOT_DIRS = configs['datasets']

    # # Create model and compile
    model = create_pointnet(configs['sampling_points'], len(configs['joints']))
    model.load_weights(configs['checkpoint_dir'])
    model.compile(
        loss=configs['loss'],
        optimizer=keras.optimizers.Adam(learning_rate=configs['learning_rate']),
        metrics=percentual_correct_keypoints(configs['threshold'])
    )

    # # Import dataset
    test_dataset = KinectDataset(
        master_root_dirs=TEST_ROOT_DIRS, 
        batch_size=configs['batch_size'],
        number_of_points=configs['sampling_points'],
        joints=configs['joints']
    )

    test_loss, test_metric = model.evaluate(test_dataset(), steps=test_dataset.dataset_size)

    logging.info("Mean loss, Mean PCK:" + str(test_loss) + ', ' + str(test_metric))
    print("Mean loss, Mean PCK:" + str(test_loss) + ', ' + str(test_metric))
