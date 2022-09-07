import os
import glob
import wandb
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from evaluate import TEST_ROOT_DIRS
from models.pointnet import create_pointnet
from datasets.kinect_dataset import KinectDataset
from metrics.metric import percentual_correct_keypoints
from wandb.keras import WandbCallback
from configs.argparser import parse_args 
from options.normalization import normalization_options
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

MASTER_ROOT_DIRS = glob.glob('D:/azure_kinect/train/*/*/master_1/')
MASTER_ROOT_DIRS = [dir for dir in MASTER_ROOT_DIRS if dir.split(os.path.sep)[-3].find('_') < 0 ]
TEST_ROOT_DIRS = glob.glob('D:/azure_kinect/test/*/*/master_1/')
VAL_ROOT_DIRS = glob.glob('D:/azure_kinect/val/*/*/master_1/')


if __name__ == '__main__':

    # Import dataset
    train_dataset = KinectDataset(
        master_root_dirs=MASTER_ROOT_DIRS, 
        number_of_points=configs['sampling_points'],
        joints=configs['joints'],
        flag='train'
    )

    val_dataset = KinectDataset(
        master_root_dirs=VAL_ROOT_DIRS, 
        number_of_points=configs['sampling_points'],
        joints=configs['joints'],
        flag='val'
    )

    test_dataset = KinectDataset(
        master_root_dirs=TEST_ROOT_DIRS, 
        number_of_points=configs['sampling_points'],
        joints=configs['joints'],
        flag='test'
    )
    
    train_ds = train_dataset().batch(configs['batch_size']).prefetch(1)
    val_ds = val_dataset().batch(configs['batch_size']).prefetch(1)
    test_ds = test_dataset().batch(configs['batch_size']).prefetch(1)

    if configs['normalization'] != '':
        train_ds = train_ds.map(normalization_options[configs['normalization']])
        val_ds = val_ds.map(normalization_options[configs['normalization']])
        test_ds = test_ds.map(normalization_options[configs['normalization']])    
        
    # Create model and compile
    model = create_pointnet(configs['sampling_points'], len(configs['joints']))

    model.compile(
        loss=configs['loss'],
        optimizer=keras.optimizers.Adam(learning_rate=configs['learning_rate']),
        metrics=percentual_correct_keypoints(configs['threshold'])
    )

    # Passing configs to wandb
    wandb.init(project=configs['project'], name=configs['name'])
    wandb.config = configs
    wandb.config['train_dir'] = MASTER_ROOT_DIRS 
    wandb.config['val_dir'] = VAL_ROOT_DIRS 

    ckpt_dir = os.path.join(configs['checkpoint_dir'], configs['project'], configs['name'])

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

    test_loss, test_metric = model.evaluate(test_ds, steps=test_dataset.dataset_size)

    logging.info("Mean loss, Mean PCK:" + str(test_loss) + ', ' + str(test_metric))
    print("Mean loss, Mean PCK:" + str(test_loss) + ', ' + str(test_metric))

    wandb.config['test loss'] = test_loss
    wandb.config['test metric'] = test_metric
    wandb.log(configs)
