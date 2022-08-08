import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import open3d as o3d
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from models.pointnet import tnet, conv_bn, dense_bn
from datasets.kinect_dataset import KinectDataset
from metrics.metric import percentual_correct_keypoints

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


MASTER_ROOT_DIRS = [
    'D:/azure_kinect/1E2DB6/02/master_1/',
    'D:/azure_kinect/4AD6F3/01/master_1/', 'D:/azure_kinect/4AD6F3/02/master_1/',
    'D:/azure_kinect/4B8AF1/01/master_1/', 'D:/azure_kinect/4B8AF1/02/master_1/',
    'D:/azure_kinect/5E373E/01/master_1/', 'D:/azure_kinect/5E373E/02/master_1/',
    'D:/azure_kinect/20E29D/01/master_1',
    'D:/azure_kinect/37A7AA/01/master_1', 
    'D:/azure_kinect/76ABFD/01/master_1', 
    'D:/azure_kinect/339F94/01/master_1', 
    'D:/azure_kinect/471EF1/01/master_1',
    'D:/azure_kinect/857F1E/01/master_1', 'D:/azure_kinect/857F1E/03/master_1', 
    'D:/azure_kinect/927394/01/master_1',
    'D:/azure_kinect/AEBA3A/01/master_1',
    'D:/azure_kinect/AFCD31/01/master_1',
    'D:/azure_kinect/C47EFC/01/master_1',
    'D:/azure_kinect/CCB8AD/01/master_1',
    'D:/azure_kinect/EEFE6D/01/master_1', 
    'D:/azure_kinect/F205FE/01/master_1', 'D:/azure_kinect/F205FE/02/master_1', 
    ]

BATCH_SIZE = 16
EPOCHS = 50
JOINTS = ['FOOT_LEFT', 'FOOT_RIGHT', 
          'ANKLE_LEFT', 'ANKLE_RIGHT',
          'KNEE_LEFT', 'KNEE_RIGHT',
          'HIP_LEFT', 'HIP_RIGHT',
          'PELVIS']
NUMBER_OF_JOINTS = len(JOINTS)
NUMBER_OF_POINTS = 1024*4
OUTPUT_TYPES = (tf.float32, tf.float32)


if __name__ == '__main__':

    # Import dataset
    kinect_dataset = KinectDataset(master_root_dirs=MASTER_ROOT_DIRS, 
                               batch_size=BATCH_SIZE, 
                               number_of_points=NUMBER_OF_POINTS,
                               joints=JOINTS)
    dataset = kinect_dataset.dataset
    train_ds, test_ds, val_ds = kinect_dataset.split_train_test_val(train_size=0.7, val_size=0.15, test_size=0.15)
    
    # Create model
    inputs = keras.Input(shape=(NUMBER_OF_POINTS, 3))

    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)

    NUM_CLASSES = kinect_dataset.number_of_joints*3
    outputs = layers.Dense(NUM_CLASSES)(x)
    model = keras.Model(inputs=inputs, 
                        outputs=outputs, 
                        name="pointnet")

    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.Adam(learning_rate=0.005),
                  metrics=[percentual_correct_keypoints])


    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints', 
                                                     verbose=1, 
                                                     save_weights_only=True,
                                                     save_freq=int((0.7*kinect_dataset.dataset_size)//BATCH_SIZE*5))
    model.fit(train_ds, 
          epochs=15, 
          validation_data=val_ds, 
          steps_per_epoch=(0.7*kinect_dataset.dataset_size)//BATCH_SIZE,
          callbacks=cp_callback)
