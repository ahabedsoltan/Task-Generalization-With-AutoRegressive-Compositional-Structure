import os

from data import *
import numpy as np
import argparse
import copy
import random

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default = '', type = str)
    args = parser.parse_args()
    config = eval(open(args.config_path, 'r').read())
    n_samples = config['n_samples']
    n_digit = config['n_lines']
    dataset_type = eval(config['dataset_type'])
    config['split'] = 'train'
    train_dataset = dataset_type(config) # change to a dictionary.
    val_config = copy.deepcopy(train_dataset.kwargs)
    val_config['split'] = 'test'
    val_config['n_samples'] = config['val_samples']
    val_dataset = dataset_type(val_config)
    name = train_dataset.get_name()
    print('Input:')
    print((train_dataset.input_ids[0]))
    PATH = "data"
    os.makedirs(f'{PATH}/Composition/{name}', exist_ok=True)
    train_dataset.save(f'{PATH}/Composition/{name}/')
    val_dataset.save(f'{PATH}/Composition/{name}/val')