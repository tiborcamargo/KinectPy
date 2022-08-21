import sys, os
import yaml
import argparse
sys.path.append('..')
from options.joints import POSSIBLE_JOINTS


def parse_args(print_config: bool = True) -> dict:
    """ 
    Argument parser that receives a config file but can be overriden
    by arguments passed via command line.
    """
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    
    parser.add_argument(
        '--config_file',
        help='configuration file *.yaml',
        default='configs/config.yaml'
    )
    
    parser.add_argument(
        '--joints',
        type=str,
        nargs='+',
    )
    
    # wandb parameters - reproducibility purposes
    parser.add_argument(
        '-p', '--project',
        help='project name for grouping experiments',
        type=str,
    )
    
    parser.add_argument(
        '-n', '--name',
        help='experiment name for reproducibility purposes',
        type=str,
    )

    parser.add_argument(
        '-d', '--train_dataset',
        help='list of root directories, eg: path/to/master_1, which contains filtered_and_registered_pointclouds dir',
        type=str,
        nargs='+',
    )

    parser.add_argument(
        '--test_dataset',
        help='list of test root directories, eg: path/to/master_1, which contains filtered_and_registered_pointclouds dir',
        type=str,
        nargs='+',
    )
    
    parser.add_argument(
        '--sampling_points',
        help='number of points to be randomly sampled in each point cloud',
        type=int,
    )
    
    parser.add_argument(
        '--train_size', 
        help='percentage of dataset to be used as training',
        type=float,
    )
    
    parser.add_argument(
        '--val_size', 
        help='percentage of dataset to be used as validation',
        type=float,
    )
    
    parser.add_argument(
        '--test_size', 
        help='percentage of dataset to be used as test',
        type=float,
    )
    
    # training parameters
    parser.add_argument(
        '-e', '--epochs', 
        help='number of epochs for training',
        type=int,
    )
    
    parser.add_argument(
        '-b', '--batch_size',
        help='number of samples per batch',
        type=int,
    )
    
    parser.add_argument(
        '-l', '--learning_rate',
        help="initial learning rate", 
        type=float, 
        required=False, 
    )
    
    parser.add_argument(
        '--loss',
        help='which loss function',
        type=str,
    )
    
    parser.add_argument(
        '-m', '--metrics',
        help='which metrics to use',
        type=str,
    )
    
    parser.add_argument(
        '-t', '--threshold',
        help='thresholds to be assessed when using PCK as metric',
        nargs='+',
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        help='root directory to save trained model',
        type=str
    )
    
    parser.add_argument(
        '--logs_dir',
        help='root directory to save all log files',
        type=str
    )
    
    args = parser.parse_args()
    args_dict = vars(args)
    
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    for key, val in args_dict.items():
        if val != None:
            config[key] = val

    # Verifying if joints are passed correctly
    for joint in config['joints']:
        if joint not in POSSIBLE_JOINTS:
            raise ValueError(f'{joint} not in {POSSIBLE_JOINTS}')

    if print_config:
        print_arguments(config)

    return config


def print_arguments(config: dict):
    print_args = 100*'-' + '\n'
    for key, value in config.items():
        print_args += '{:20s} {:20s}\n'.format(str(key), str(value))
    print_args += 100*'-' + '\n'
    print(print_args)
    return print_args
