import argparse
from options.joints import LOWER_JOINTS
from yacs.config import CfgNode as CN

_C = CN()

# Project information
_C.PROJECT = CN()
_C.PROJECT.NAME = 'sampling_points_experiments'
_C.PROJECT.EXPERIMENT = 'yacs'

# Saving locations
_C.DIRS = CN()
_C.DIRS.CHECKPOINT_DIR = './checkpoints'
_C.DIRS.LOGS_DIR = './logs'

# Data parameters
_C.DATA = CN()
_C.DATA.JOINTS = LOWER_JOINTS
_C.DATA.SAMPLING_POINTS = 1024
_C.DATA.NORMALIZATION = ''

# Training parameters
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 30
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.LEARNING_RATE = 0.005
_C.TRAIN.LOSS = 'mean_squared_error'

# Evaluation parameters
_C.EVAL = CN()
_C.EVAL.METRICS = 'percentual_correct_keypoints'
_C.EVAL.THRESHOLDS = 50


def get_default_config(parser):
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  args = parser.parse_args()

  if args.config:
      _C.merge_from_file(args.config)
  _C.freeze()
  return _C.clone()


# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`

if __name__ == '__main__':
  configs = get_default_config()
  print(configs)
  print('---------')
  print(configs.TRAIN.EPOCHS)
