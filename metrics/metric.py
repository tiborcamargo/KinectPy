import numpy as np
import tensorflow as tf
from typing import Union


def percentual_correct_keypoints(y_true, y_pred, threshold=150):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    numerator = tf.where(tf.abs(y_pred - y_true) <= threshold, x=1.0, y=0.0)
    value = tf.math.reduce_mean(numerator)
    return value

#class PercentualCorrectKeypoints(tf.keras.metrics.Metric):
#    """
#    Computes the PCK metric using a user-defined threshold. 
#    """
#    def __init__(
#        self, 
#        threshold: int = 150, 
#        name: str = 'percentual_correct_keypoints', 
#        **kwargs
#        ):
#        """
#        Args:
#            threshold: Threshold in milimeters for assigning a correct prediction
#        """
#        super(PercentualCorrectKeypoints, self).__init__(name=name, **kwargs)
#        self.threshold = threshold 
#        self.result_accumulator = self.add_weight(name='result', initializer='zeros', dtype=tf.float32)
#        self.counter = self.add_weight(name='counter', initializer='zeros', dtype=tf.int32)
#    
#    
#    def update_state(self, y_true: Union[tf.float, np.ndarray], y_pred: Union[tf.float, np.ndarray]):
#        y_true = tf.cast(y_true, tf.float32)
#        y_pred = tf.cast(y_pred, tf.float32)
#        numerator = tf.where(tf.abs(y_pred - y_true) <= self.threshold, x=1.0, y=0.0)
#        value = tf.math.reduce_mean(numerator)
#        self.counter.assign_add(1)
#        self.result_accumulator.assign_add(value)
#
#        
#    def result(self):
#        return self.result_accumulator/tf.cast(self.counter, tf.float32)
