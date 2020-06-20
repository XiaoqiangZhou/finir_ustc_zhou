import os
import csv
import time
import argparse
import numpy as np
import multiprocessing
from functools import partial
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

from toy_experiments import read_csv, cal_acc

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Script to run')
    parser.add_argument('-v', '--version', type=str, default='v1')
    parser.add_argument('-n', '--exp_name', type=str, default='debug')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    return args

class RNN(nn.Module):
    def __init__(self, LOOK_BACK, hide_dim=6, layer_num=2):
        super(RNN,self).__init__()
        self.lstm = nn.LSTM(LOOK_BACK, hide_dim, layer_num)
        self.out = nn.Linear(hide_dim, 1)

    def forward(self,x):
        x1,_=self.lstm(x)
        a,b,c = x1.shape
        out = self.out(x1.view(-1,c))
        out1 = out.view(a,b,-1)
        return out1

def create_dataset(dataset, look_back, task_day):
    data_x = []
    data_y = []
    for i in range(len(dataset) - (look_back+task_day)+1):
        data_x.append(dataset[i:i+look_back])          # diff
        data_y.append(dataset[i+look_back+task_day-1])
    return np.asarray(data_x), np.asarray(data_y)

def new_create_dataset(dataset, look_back, task_day):
    data_x = []
    data_y = []
    for i in range(len(dataset) - (look_back*task_day)+1):
        temp = []
        for j in range(look_back):
            temp.append(dataset[i+j*task_day-1])
        data_x.append(temp)
        data_y.append(dataset[i+(look_back-1)*task_day+task_day-1])
    return np.asarray(data_x), np.asarray(data_y)

def train_a_regressor(metal_name, task_day, exp_name='demo', look_back=2, train_length=200, learn_rate=0.02, epochs=500, train=True, hide_dim=6, layer_num=2, version='v1'):
    """
    Train a LSTM.
    task day is 1 or 20 or 60
    """
    ### data preparation
    ## read files
    # 3M data
    train_data_root = os.path.join('data', 'Train', 'Train_data')
    val_data_root = os.path.join('data', 'Validation', 'Validation_data')
    train_3M_file = os.path.join(train_data_root, 'LME'+metal_name+'3M_train.csv')
    val_3M_file = os.path.join(val_data_root, 'LME'+metal_name+'3M_validation.csv')
    
    train_length=train_length+task_day*(look_back+1)-1
    train_3M_data = read_csv(train_3M_file)
    train_close_price = np.array([float(_) for _ in train_3M_data['Close.Price']][-train_length:])# !
    val_3M_data = read_csv(val_3M_file)
    val_close_price = np.array([float(_) for _ in val_3M_data['Close.Price']])# !
    
    # train label data (not necessary)
    train_label_taskday_file = os.path.join(train_data_root, 'Label_LME'+metal_name+'_train_'+str(task_day)+'d.csv')
    label_taskday_data = read_csv(train_label_taskday_file)
    for key, value in label_taskday_data.items():
        if key.startswith(u'LM'):
            KEY = key
            break
    label_taskday = np.array([float(_) for _ in label_taskday_data[KEY]])[-train_length:] # !
    
    ## preprocess data
    train_data_min = train_close_price.min() # diff, only training data
    train_data_max = train_close_price.max()
    
    train_data = train_close_price
    train_data = (train_data - train_data_min) / (train_data_max - train_data_min)
    train_label = []
    if version == 'v1':
        train_dataX, train_dataY = create_dataset(train_data, look_back * task_day, task_day)
        train_dataX = train_dataX.reshape(-1,1,look_back * task_day)
    elif version == 'v2':
        train_dataX, train_dataY = new_create_dataset(train_data, look_back, task_day)
        train_dataX = train_dataX.reshape(-1,1,look_back)
    train_dataY = train_dataY.reshape(-1,1,1)
    train_dataX = torch.from_numpy(train_dataX)
    train_dataY = torch.from_numpy(train_dataY)

    ### model training
    if version == 'v1':
        model = RNN(LOOK_BACK=look_back * task_day, hide_dim=hide_dim, layer_num=layer_num)
    elif version == 'v2':
        model = RNN(LOOK_BACK=look_back, hide_dim=hide_dim, layer_num=layer_num)
    if train:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate)
        loss_func = nn.MSELoss()
        for i in range(epochs):
            var_x = Variable(train_dataX).type(torch.FloatTensor)
            var_y = Variable(train_dataY).type(torch.FloatTensor)
            if torch.cuda.is_available():
                var_x = var_x.cuda()
                var_y = var_y.cuda()
                loss_func = loss_func.cuda()
                model = model.cuda()
            out = model(var_x)
            loss = loss_func(out,var_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1)%200 == 0:
                print('Epoch:{},loss:{:.5f}'.format(i+1,loss.item()))
        torch.save(model.state_dict(), os.path.join('models', exp_name, metal_name+"_"+str(task_day)+'day.pth'))
    else:
        model.eval()
        ckpt = torch.load(os.path.join('models', exp_name, metal_name+"_"+str(task_day)+'day.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(ckpt)

    ### model inference
    model.eval()
    # 17年最后一天，18年第一天  -> 18年第(1+k)天的预测目标值
    start_point = 0# -(task_day*(look_back+1)-1) # or 0
    val_close_price = np.append(val_close_price, np.array([train_data_min]*task_day))  # !
    test_data = np.concatenate((train_close_price[start_point:], val_close_price))
    test_data = (test_data - train_data_min) / (train_data_max - train_data_min)
    if version == 'v1':
        test_dataX, test_dataY = create_dataset(test_data, look_back * task_day, task_day)
        test_dataX = test_dataX.reshape(-1, 1, look_back * task_day)
    elif version == 'v2':
        test_dataX, test_dataY = new_create_dataset(test_data, look_back, task_day)
        test_dataX = test_dataX.reshape(-1, 1, look_back)
    test_label = []
    
    test_dataX = torch.from_numpy(test_dataX)
    test_dataX = Variable(test_dataX).type(torch.FloatTensor)

    ## 可用的数据
    with torch.no_grad():
        model = model.to('cpu') if torch.cuda.is_available() else model
        pred_test = model(test_dataX)
    pred_test = pred_test.view(-1).data.numpy()

    ### results saving
    output_file = os.path.join('results', exp_name, metal_name+'_'+str(task_day)+'day.csv')
    with open(output_file, 'w', newline='') as f:
        pred_reversed = list(pred_test)[::-1]
        # import ipdb; ipdb.set_trace()
        val_close_price = (val_close_price - train_data_min)/(train_data_max - train_data_min)
        test_dataY_reversed = val_close_price[::-1]
        flags = []
        for i in range(253):
            flag = 1 if pred_reversed[i]-test_dataY_reversed[i]>=0 else 0
            flags.append(flag)
        flags = flags[::-1]
        writer = csv.writer(f)
        for i in range(253):
            writer.writerow([str(task_day)+'_day', int(flags[i])])
    

if __name__ == "__main__":
    exp_name = 'v2'
    os.makedirs(os.path.join('results', exp_name), exist_ok=True)
    os.makedirs(os.path.join('models', exp_name), exist_ok=True)
    taskdays = [1,20,60]
    metal_names = ['Lead', 'Nickel', 'Tin', 'Zinc', 'Copper', 'Aluminium']
    train=True
    if 0:
        for metal_name in metal_names:
            for taskday in taskdays:
                train_a_regressor(metal_name=metal_name, task_day=taskday, train=train, exp_name=exp_name)
    elif 1: # grid search
        # look_backs=[2,4]
        # train_lengths=[400, 600, 800]
        # learn_rates=[0.02]
        # epochses=[600, 800, 1000]
        # hide_dims=[16]
        # layer_nums=[2]
        look_backs=[4]
        train_lengths=[800]
        learn_rates=[0.02]
        epochses=[600]
        hide_dims=[128] # 4->16 , 80->320
        layer_nums=[2]
        for look_back in look_backs:
            for train_length in train_lengths:
                for learn_rate in learn_rates:
                    for epochs in epochses:
                        for hide_dim in hide_dims:
                            for layer_num in layer_nums:
                                exp_name_str = 'look_back_'+str(look_back)+'_train_length_'+str(train_length)+'_learn_rate_'+str(learn_rate)+'_epochs_'+str(epochs)+'_hide_dim_'+str(hide_dim)+'_layer_num_'+str(layer_num)
                                exp_name = os.path.join('grid_search', exp_name_str)
                                os.makedirs(os.path.join('results', exp_name), exist_ok=True)
                                os.makedirs(os.path.join('models', exp_name), exist_ok=True)
                                for metal_name in metal_names:
                                    for taskday in taskdays:
                                        train_a_regressor(metal_name=metal_name, 
                                                          task_day=taskday, 
                                                          look_back=look_back,
                                                          train_length=train_length,
                                                          learn_rate=learn_rate,
                                                          epochs=epochs,
                                                          hide_dim=hide_dim,
                                                          layer_num=layer_num,
                                                          exp_name=exp_name)
    elif 1:
        args = parse_args()
        train = not args.test
        version = args.version
        exp_name = args.exp_name
        os.makedirs(os.path.join('results', exp_name), exist_ok=True)
        os.makedirs(os.path.join('models', exp_name), exist_ok=True)
        for metal_name in metal_names:
            for taskday in taskdays:
                train_a_regressor(metal_name=metal_name, 
                                task_day=taskday, 
                                look_back=4,
                                train_length=400,
                                train=train,
                                exp_name=exp_name,
                                epochs=600,
                                hide_dim=16,
                                version=version)

    if 0: # caution
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=3)
        params = [(x,y) for x in metal_names for y in taskdays]
        pool.starmap(train_a_regressor,params)
        

# TODO: 绘制均线图成交量，换用其他模型，回归->二分类任务，long term+short term, transformer, 动态区间, 统计测试集涨跌分布
# [1,2,3,4,5] --> [25]
# [[1,2,3,4,5], [-5,1,5]] --> [25]
# [N, 20, 1]