import numpy as np
import tensorflow as tf
from typing import Union


def percentual_correct_keypoints(threshold=150):
    def PercentualCorrectKeypoints(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        numerator = tf.where(tf.abs(y_pred - y_true) <= threshold, x=1.0, y=0.0)
        value = tf.math.reduce_mean(numerator)
        return value
    return PercentualCorrectKeypoints
