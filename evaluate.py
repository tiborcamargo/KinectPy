import os
import csv
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from models.pointnet import create_pointnet
from datasets.kinect_dataset import KinectDataset
from metrics.metric import percentual_correct_keypoints
from configs.argparser import parse_args 
from options.normalization import normalization_options
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

    if configs['test_dataset']:
        TEST_ROOT_DIRS = configs['test_dataset']

    # Create model and compile
    model = create_pointnet(configs['sampling_points'], len(configs['joints']))

    print(configs['checkpoint_dir'])
    model.load_weights(configs['checkpoint_dir'])
    model.compile(
        loss=configs['loss'],
        optimizer=keras.optimizers.Adam(learning_rate=configs['learning_rate']),
        metrics=percentual_correct_keypoints(configs['threshold'])
    )

    # Import dataset
    test_dataset = KinectDataset(
        master_root_dirs=TEST_ROOT_DIRS, 
        batch_size=configs['batch_size'],
        number_of_points=configs['sampling_points'],
        joints=configs['joints']
    )

    if configs['normalization'] != '':
        test_dataset_size = test_dataset.dataset_size
        test_dataset = test_dataset().map(normalization_options[configs['normalization']])

        test_loss, test_metric = model.evaluate(test_dataset, steps=test_dataset_size)
    else:
        test_loss, test_metric = model.evaluate(test_dataset(), steps=test_dataset.dataset_size)

    # Evaluate

    # Save and print evaluation
    result_message = f"PCK@{configs['threshold']}: Mean loss = {str(test_loss)}, Mean PCK = {str(test_metric)}" 
    evaluation_fp = os.path.join(
        configs['checkpoint_dir'], 
        f"evaluation_{configs['project']}_{configs['name']}.txt"
        )

    with open (evaluation_fp, 'a') as filedata:                            
        filedata.write(result_message + '\n')

    logging.info(result_message)
    print(result_message)
