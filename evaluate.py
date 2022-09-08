import os
import glob
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from models.pointnet import create_pointnet
from datasets.kinect_dataset import KinectDataset
from metrics.metric import percentual_correct_keypoints
from configs.argparser import parse_args 
from options.normalization import normalization_options
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.set_printoptions(suppress=True)
tf.random.set_seed(1234)


TEST_ROOT_DIRS = glob.glob('D:/azure_kinect/test/*/*/master_1')


if __name__ == '__main__':

    # Read arguments
    configs = parse_args(print_config=True)

    # Create model and compile
    model = create_pointnet(configs['sampling_points'], len(configs['joints']))

    checkpoint_dir = os.path.join(configs['checkpoint_dir'], configs['project'], configs['name'])
    logging.info(f'Loading from ckpt from {checkpoint_dir}')
    model.load_weights(checkpoint_dir)
    model.compile(
        loss=configs['loss'],
        optimizer=keras.optimizers.Adam(learning_rate=configs['learning_rate']),
        metrics=percentual_correct_keypoints(configs['threshold'])
    )

    # Import dataset
    test_dataset = KinectDataset(
        master_root_dirs=TEST_ROOT_DIRS, 
        number_of_points=configs['sampling_points'],
        joints=configs['joints'],
        flag='test'
    )
    
    test_ds = test_dataset().batch(configs['batch_size']).prefetch(1)

    if configs['normalization'] != '':
        test_ds = test_ds.map(normalization_options[configs['normalization']])    
        test_dataset_size = test_dataset.dataset_size
        test_loss, test_metric = model.evaluate(test_ds, steps=test_dataset_size//configs['batch_size'])
    else:
        test_loss, test_metric = model.evaluate(test_ds, steps=test_dataset.dataset_size//configs['batch_size'])

    # Save and print evaluation
    result_message = f"PCK@{configs['threshold']}: Mean loss = {str(test_loss)}, Mean PCK = {str(test_metric)}" 
    evaluation_fp = os.path.join(
        checkpoint_dir,
        f"evaluation.txt"
        )

    with open (evaluation_fp, 'a') as filedata:                            
        filedata.write(result_message + '\n')
        logging.info('Data saved at:', evaluation_fp)

    logging.info(result_message)
