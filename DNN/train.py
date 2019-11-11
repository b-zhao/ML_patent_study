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
from loader import  PatentDataset, load_data_train, load_data_with_convert_Y
from DNN.model import Net, Net_with_softmax
from mxnet import autograd, gluon, init, nd
from sklearn.decomposition import PCA



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



    if opt.convert_y:
        xtrain, ytrain, xtest, ytest = load_data_with_convert_Y()
    else:
        xtrain, ytrain, xtest, ytest = load_data_train()





    datasets = {}
    datasets['train'] = PatentDataset(xtrain, ytrain)
    datasets['val'] = PatentDataset(xtest, ytest)

    N_input = xtrain.shape[1]
    N_output = ytrain.shape[1]

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=opt.batchsize,
                                                  shuffle=True, num_workers=8, pin_memory=True)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    use_gpu = torch.cuda.is_available()



    model =  Net_with_softmax(N_input, 1024, N_output) if opt.convert_y else Net(N_input, 1024, N_output)

    # if opt.model_dir and opt.train_on_save:
    #     model.load_state_dict(torch.load(opt.model_dir))

    model = model.double().cuda()

    optimizer_adam = torch.optim.Adam(model.parameters(), lr = opt.lr)

    loss_function = torch.nn.MSELoss()


    def train_model(model, criterion, optimizer, num_epochs=25):
        train_loss = []
        test_loss = []
        accuracy_train = []
        accuracy_test = []

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            for phase in ['train', 'val']:

                if phase == 'train':
                    model.train(True)
                else:
                    model.train(False)
                running_loss = 0.0
                totalNum = 0
                true_predict = 0
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

                    predict_result = outputs.cpu().detach().numpy().reshape(-1)
                    ground_truth = labels.cpu().detach().numpy().reshape(-1)

                    true_predict += np.where( abs(predict_result - ground_truth) < 100)[0].shape[0]

                    loss = criterion(outputs, labels)
                    running_loss += loss
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    pbar.set_description(desc='loss: {:.4f}, accuracy: {:.4f}'.format(running_loss /totalNum, true_predict / totalNum))
                if phase == 'train':
                    train_loss.append(loss.cpu().detach().numpy() / totalNum)
                    accuracy_train.append(true_predict / totalNum)
                else:
                    test_loss.append(loss.cpu().detach().numpy() / totalNum)
                    accuracy_test.append(true_predict / totalNum)

        train_loss = np.array(train_loss)
        test_loss = np.array(test_loss)
        accuracy_train = np.array(accuracy_train)
        accuracy_test = np.array(accuracy_test)

        np.save('train_loss.npy', train_loss)
        np.save('test_loss.npy', test_loss)
        np.save('train_accuracy.npy', accuracy_train)
        np.save('test_accuracy.npy', accuracy_test)

        # do final operation here
        torch.save(model.state_dict(), './models/model_%s_epoch_%s.pkl' % (datetime.datetime.now().strftime("%Y_%m_%d_%H"), num_epochs))
    print(opt)
    train_model(model, loss_function, optimizer_adam, num_epochs=opt.epoch)