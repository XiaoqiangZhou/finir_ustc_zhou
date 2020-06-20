import os
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from toy_experiments import read_csv, cal_acc
from rnn import train_a_regressor

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Script to reimplement our result')
    parser.add_argument('-n', '--exp_name', type=str, default='debug', help='first/second/third_task')
    parser.add_argument('-t', '--test', action='store_true', help='retrain the model or use pretrained model?')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    train = not args.test
    exp_name = args.exp_name
    os.makedirs(os.path.join('results', exp_name), exist_ok=True)
    os.makedirs(os.path.join('models', exp_name), exist_ok=True)
    taskdays = [1,20,60]
    metal_names = ['Lead', 'Nickel', 'Tin', 'Zinc', 'Copper', 'Aluminium']

    if exp_name in ['first_task', 'first_task_pretrained']:
        look_back = 4
        train_length = 800
        learn_rate = 0.02
        epochs = 600
        hide_dim = 16
        layer_num = 2
        version = 'v1'
        for metal_name in metal_names:
            for task_day in taskdays:
                train_a_regressor(metal_name=metal_name, task_day=task_day, exp_name=exp_name, look_back=look_back, train_length=train_length, learn_rate=learn_rate, epochs=epochs, train=train, hide_dim=hide_dim, layer_num=layer_num, version=version)
    elif exp_name in ['second_task', 'second_task_pretrained']:
        look_back = 4
        train_length = 400
        epochs = 600
        hide_dim = 32
        learn_rate = 0.02
        layer_num = 2
        version = 'v2'
        for metal_name in metal_names:
            for task_day in taskdays:
                train_a_regressor(metal_name=metal_name, task_day=task_day, exp_name=exp_name, look_back=look_back, train_length=train_length, learn_rate=learn_rate, epochs=epochs, train=train, hide_dim=hide_dim, layer_num=layer_num, version=version)
    elif exp_name in ['third_task', 'third_task_pretrained']:
        look_back = 4
        train_length = 400
        epochs = 600
        hide_dim = 16
        learn_rate = 0.02
        layer_num = 2
        version = 'v2'
        for metal_name in metal_names:
            for task_day in taskdays:
                train_a_regressor(metal_name=metal_name, task_day=task_day, exp_name=exp_name, look_back=look_back, train_length=train_length, learn_rate=learn_rate, epochs=epochs, train=train, hide_dim=hide_dim, layer_num=layer_num, version=version)

