import wandb
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import open3d as o3d
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from models.pointnet import create_pointnet
from datasets.kinect_dataset import KinectDataset
from metrics.metric import percentual_correct_keypoints
from wandb.keras import WandbCallback

tf.random.set_seed(1234)

try:
    os.mkdir('logs')
except:
    pass

logging.basicConfig(filename='logs/train.log', 
                    level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    force=True)

logger = logging.getLogger(__name__)
np.set_printoptions(suppress=True)
tf.random.set_seed(1234)
# TO-DO: Change the Project name using some argument parser
wandb.init(project="KinectPy", name="first_try")


MASTER_ROOT_DIRS = [
    'D:/azure_kinect/1E2DB6/02/master_1/',
    'D:/azure_kinect/4AD6F3/01/master_1/', 'D:/azure_kinect/4AD6F3/02/master_1/',
    'D:/azure_kinect/4B8AF1/01/master_1/', 'D:/azure_kinect/4B8AF1/02/master_1/',
    #'D:/azure_kinect/5E373E/01/master_1/', 'D:/azure_kinect/5E373E/02/master_1/',
    #'D:/azure_kinect/20E29D/01/master_1',
    #'D:/azure_kinect/37A7AA/01/master_1', 
    #'D:/azure_kinect/76ABFD/01/master_1', 
    #'D:/azure_kinect/339F94/01/master_1', 
    #'D:/azure_kinect/471EF1/01/master_1',
    #'D:/azure_kinect/857F1E/01/master_1', 'D:/azure_kinect/857F1E/03/master_1', 
    #'D:/azure_kinect/927394/01/master_1',
    #'D:/azure_kinect/AEBA3A/01/master_1',
    #'D:/azure_kinect/AFCD31/01/master_1',
    #'D:/azure_kinect/C47EFC/01/master_1',
    #'D:/azure_kinect/CCB8AD/01/master_1',
    #'D:/azure_kinect/EEFE6D/01/master_1', 
    #'D:/azure_kinect/F205FE/01/master_1', 'D:/azure_kinect/F205FE/02/master_1', 
    ]

EPOCHS = 25
BATCH_SIZE = 16
TRAIN_SIZE = 0.9
VAL_SIZE = 0.1
TEST_SIZE = 0
STARTING_LR = 5e-3
JOINTS = ['FOOT_LEFT', 'FOOT_RIGHT', 
          'ANKLE_LEFT', 'ANKLE_RIGHT',
          'KNEE_LEFT', 'KNEE_RIGHT',
          'HIP_LEFT', 'HIP_RIGHT',
          'PELVIS']
NUMBER_OF_JOINTS = len(JOINTS)
NUMBER_OF_POINTS = 128
THRESHOLD = 50
OUTPUT_TYPES = (tf.float32, tf.float32)

wandb.config = {
    "learning_rate": STARTING_LR,
    "sampling_points": NUMBER_OF_POINTS,
    "number_of_joints": NUMBER_OF_JOINTS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "threshold_pck": THRESHOLD,
    "joints": JOINTS,
    "train_size": TRAIN_SIZE,
    "val_size": VAL_SIZE,
    "test_size": TEST_SIZE,
    "datasets": MASTER_ROOT_DIRS,
}

if __name__ == '__main__':

    # Import dataset
    kinect_dataset = KinectDataset(master_root_dirs=MASTER_ROOT_DIRS, 
                                   batch_size=BATCH_SIZE, 
                                   number_of_points=NUMBER_OF_POINTS,
                                   joints=JOINTS)
    dataset = kinect_dataset.dataset
    train_ds, test_ds, val_ds = kinect_dataset.split_train_test_val(train_size=TRAIN_SIZE, 
                                                                    val_size=VAL_SIZE, 
                                                                    test_size=TEST_SIZE)
    
    # Create model
    model = create_pointnet(NUMBER_OF_POINTS, NUMBER_OF_JOINTS)

    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.Adam(learning_rate=STARTING_LR),
                  metrics=[percentual_correct_keypoints(threshold=THRESHOLD)])


    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints', 
                                                     verbose=1, 
                                                     save_weights_only=True,
                                                     save_freq='epoch')


    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

    train_ds = train_ds.repeat()

    model.fit(train_ds, 
              epochs=EPOCHS, 
              validation_data=val_ds, 
              steps_per_epoch=(0.7*kinect_dataset.dataset_size)//BATCH_SIZE,
              validation_steps=(0.15*kinect_dataset.dataset_size)//BATCH_SIZE,
              callbacks=[cp_callback, lr_callback, WandbCallback()])

    wandb.tensorflow.log(tf.summary.create_file_writer('logs/tf_experiments'))