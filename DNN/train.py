import argparse
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils import data as Data
import torch.backends.cudnn as cudnn
import matplotlib
import logging
import csv
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas
import time
import datetime
from loader import  PatentDataset, load_data_new
from DNN.model import Net
from mxnet import autograd, gluon, init, nd


def train(opt):

    data_dir = opt.data_dir
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True




    xtrain, ytrain, xtest, ytest = load_data_new()

    datasets = {}
    datasets['train'] = PatentDataset(xtrain, ytrain)
    datasets['val'] = PatentDataset(xtest, ytest)

    N_input = xtrain.shape[1]

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=opt.batchsize,
                                                  shuffle=True, num_workers=4, pin_memory=True)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    use_gpu = torch.cuda.is_available()

    def get_net():
        net = nn.Sequential()
        net.add(nn.Dense(1))
        net.initialize()
        return net

    # def log_rmse(net, features, labels):
    #     # 将小于1的值设成1，使得取对数时数值更稳定
    #     clipped_preds = nd.clip(net(features), 1, float('inf'))
    #     rmse = nd.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    #     return rmse.asscalar()

    model = Net(N_input,1024, 1)
    # model = get_net()

    # if opt.model_dir and opt.train_on_save:
    #     model.load_state_dict(torch.load(opt.model_dir))

    model = model.double().cuda()


    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=opt.lr)  # 传入网络参数和学习率
    optimizer_adam = torch.optim.Adam(model.parameters(), lr = opt.lr)
    loss_function = torch.nn.MSELoss()  # 最小均方误差

    def train_model(model, criterion, optimizer, num_epochs=25):
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            for phase in ['train', 'val']:

                if phase == 'train':
                    model.train(True)
                else:
                    model.train(False)
                running_loss = 0.0
                totalNum = 0

                pbar = tqdm(dataloaders[phase])
                for inputs, labels in pbar:
                    batch_size, N = inputs.shape
                    totalNum += batch_size

                    optimizer.zero_grad()
                    if use_gpu:
                        inputs = inputs.double().cuda()
                        labels = labels.double().cuda()
                    if phase == 'val':
                        with torch.no_grad():
                            outputs = model(inputs)
                    else:
                        outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    pbar.set_description(desc='loss: {:.4f}'.format(running_loss / totalNum))
        # do final operation here
        torch.save(model.state_dict(), './models/model_%s_epoch_%s.pkl' % (datetime.datetime.now().strftime("%Y_%m_%d_%H"), num_epochs))
    print(opt)
    train_model(model, loss_function, optimizer_adam, num_epochs=opt.epoch)