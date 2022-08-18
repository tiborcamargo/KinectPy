import os
import wandb
import logging
import datetime
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from models.pointnet import create_pointnet
from datasets.kinect_dataset import KinectDataset
from metrics.metric import percentual_correct_keypoints
from wandb.keras import WandbCallback
from configs.argparser import parse_args 
np.set_printoptions(suppress=True)
tf.random.set_seed(1234)

configs = parse_args(print_config=True)
# Logging options
Path(configs['logs_dir']).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(configs['logs_dir'], 'train.log'), 
    level=logging.DEBUG, 
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    force=True
)

wandb.init(project=configs['project'], name=configs['name'])

# MASTER_ROOT_DIRS = [
#     'D:/azure_kinect/1E2DB6/01/master_1/', 'D:/azure_kinect/1E2DB6/02/master_1/', 'D:/azure_kinect/1E2DB6/04/master_1/', 'D:/azure_kinect/1E2DB6/05/master_1/', 'D:/azure_kinect/1E2DB6/06/master_1/',
#     'D:/azure_kinect/4AD6F3/01/master_1/', 'D:/azure_kinect/4AD6F3/02/master_1/', 'D:/azure_kinect/4AD6F3/03/master_1/', 'D:/azure_kinect/4AD6F3/04/master_1/', 'D:/azure_kinect/4AD6F3/12/master_1/',
#     'D:/azure_kinect/4B8AF1/01/master_1/', 'D:/azure_kinect/4B8AF1/02/master_1/', 'D:/azure_kinect/4B8AF1/03/master_1/', 'D:/azure_kinect/4B8AF1/04/master_1/',
#     'D:/azure_kinect/5E373E/01/master_1/', 'D:/azure_kinect/5E373E/02/master_1/', 'D:/azure_kinect/5E373E/03/master_1/',
#     'D:/azure_kinect/20E29D/01/master_1/', 'D:/azure_kinect/20E29D/02/master_1',  'D:/azure_kinect/20E29D/03/master_1', 'D:/azure_kinect/20E29D/04/master_1',
#     'D:/azure_kinect/37A7AA/02/master_1/', 'D:/azure_kinect/37A7AA/03/master_1/', 'D:/azure_kinect/37A7AA/04/master_1/',
#     'D:/azure_kinect/339F94/01/master_1', 'D:/azure_kinect/339F94/02/master_1','D:/azure_kinect/339F94/03/master_1','D:/azure_kinect/339F94/04/master_1',
#     'D:/azure_kinect/471EF1/01/master_1','D:/azure_kinect/471EF1/02/master_1','D:/azure_kinect/471EF1/04/master_1',
#     'D:/azure_kinect/857F1E/01/master_1', 'D:/azure_kinect/857F1E/03/master_1', 'D:/azure_kinect/857F1E/04/master_1',
#     'D:/azure_kinect/927394/01/master_1','D:/azure_kinect/927394/02/master_1','D:/azure_kinect/927394/03/master_1',
#     'D:/azure_kinect/AEBA3A/01/master_1',
#     'D:/azure_kinect/AFCD31/03/master_1', # all others have bad registration so far
#     'D:/azure_kinect/F205FE/01/master_1', 'D:/azure_kinect/F205FE/02/master_1/', 
# ]

MASTER_ROOT_DIRS = [
    'D:/azure_kinect/debug2/master_1'
]

# TEST_ROOT_DIRS = [
#     'D:/azure_kinect/76ABFD/03/master_1', 'D:/azure_kinect/76ABFD/04/master_1', 
#     'D:/azure_kinect/CCB8AD/01/master_1', 
# ]

if __name__ == '__main__':

    # Import dataset
    kinect_dataset = KinectDataset(
        master_root_dirs=MASTER_ROOT_DIRS, 
        batch_size=configs['batch_size'],
        number_of_points=configs['sampling_points'],
        joints=configs['joints']
    )

    dataset = kinect_dataset.dataset

    train_ds, val_ds = kinect_dataset.split_train_val(
        train_split=configs['train_size'], 
        val_split=configs['val_size'], 
    )

    # Passing configs to wandb
    wandb.config = configs
    wandb.config['dataset_fullsize'] = kinect_dataset.dataset_size
    wandb.config['file_dirs'] = MASTER_ROOT_DIRS 
    
    # Create model and compile
    model = create_pointnet(configs['sampling_points'], len(configs['joints']))

    model.compile(
        loss=configs['loss'],
        optimizer=keras.optimizers.Adam(learning_rate=configs['learning_rate']),
        metrics=percentual_correct_keypoints(configs['threshold'])
    )

    now = datetime.datetime.now()
    suffix = str(now.year) + '{:02d}'.format(now.month) + '' + '{:02d}'.format(now.day) 
    ckpt_dir = os.path.join(
        configs['checkpoint_dir'], configs['project'], configs['name'] + '_' + suffix
    )

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_dir, 
        verbose=1, 
        save_weights_only=False,
        save_freq='epoch'
        )

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

    model.fit(
        train_ds, 
        epochs=configs['epochs'], 
        validation_data=val_ds, 
        callbacks=[cp_callback, lr_callback, WandbCallback()]
    )

    wandb.tensorflow.log(
        tf.summary.create_file_writer(os.path.join(configs['logs_dir'], 'tf_experiments'))
    )

    # Import dataset
    test_dataset = KinectDataset(
        master_root_dirs=TEST_ROOT_DIRS, 
        batch_size=configs['batch_size'],
        number_of_points=configs['sampling_points'],
        joints=configs['joints']
    )

    test_loss, test_metric = model.evaluate(test_dataset(), steps=test_dataset.dataset_size)

    logging.info("Mean loss, Mean PCK:" + str(test_loss) + ', ' + str(test_metric))
    print("Mean loss, Mean PCK:" + str(test_loss) + ', ' + str(test_metric))

    wandb.config['test loss'] = test_loss
    wandb.config['test metric'] = test_metric
    wandb.log(configs)
