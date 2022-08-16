import os
import wandb
import logging
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import open3d as o3d
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
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


MASTER_ROOT_DIRS = ['D:/azure_kinect/1E2DB6/01/master_1/']


if __name__ == '__main__':

    # Import dataset
    kinect_dataset = KinectDataset(
        master_root_dirs=MASTER_ROOT_DIRS, 
        batch_size=configs['batch_size'],
        number_of_points=configs['sampling_points'],
        joints=configs['joints']
    )

    dataset = kinect_dataset.dataset

    train_ds, test_ds, val_ds = kinect_dataset.split_train_test_val(
        train_size=configs['train_size'], 
        val_size=configs['val_size'], 
        test_size=configs['test_size']
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
        configs['checkpoint_dir'], configs['name'], configs['name'] + '_' + suffix
    )

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_dir, 
        verbose=1, 
        save_weights_only=True,
        save_freq='epoch'
    )

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

    train_ds = train_ds.repeat()

    model.fit(
        train_ds, 
        epochs=configs['epochs'], 
        validation_data=val_ds, 
        steps_per_epoch=(configs['train_size']*kinect_dataset.dataset_size)//configs['batch_size'],
        validation_steps=(configs['val_size']*kinect_dataset.dataset_size)//configs['batch_size'],
        callbacks=[cp_callback, lr_callback, WandbCallback()]
    )

    wandb.tensorflow.log(
        tf.summary.create_file_writer(os.path.join(configs['logs_dir'], 'tf_experiments'))
    )

    # Save results
    val_loss = 0
    val_pck = 0
    num_of_batches = 1
    for x_test, y_test in val_ds.take(10):
        y_pred = model.predict(x_test)
        loss, pck = model.evaluate(x_test, y_test)
        val_loss += loss
        val_pck += pck
        num_of_batches += 1
    
    mean_val_loss = val_loss/num_of_batches
    mean_val_pck = val_pck/num_of_batches
    logging.info("Mean loss, Mean PCK:" + str(mean_val_loss) + ', ' + str(mean_val_pck))
