import argparse

import logging
import csv
import yaml
from DNN.train import train
from DNN.test import test
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')

parser.add_argument('--data_dir',default='../data/market/pytorch',type=str, help='training dir path')
parser.add_argument('--test_dir',default='../data/duke/pytorch',type=str, help='./test_data')

parser.add_argument('--model_dir',default='./models/2019_11_16_08/model_epoch_9.pkl',type=str, help='./model_data')
parser.add_argument('--train_on_save', default= False, action='store_true', help='use saved model training data' )

parser.add_argument('--convert_y', default=False, action='store_true', help='use converted Y' )
parser.add_argument('--PCA', default=False, action='store_true', help='use PCA to comress X ' )
parser.add_argument('--LDA', default=False, action='store_true', help='use LDA to comress X ' )

parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')

parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epoch', default=20, type=int, help='training epoch')

parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')

parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')

opt = parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(filename="running.log", level=logging.DEBUG, format='%(asctime)s %(message)s')

    file_name = './dnnResult.txt'
    with open(file_name, 'a+') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

    train(opt)
    #test(opt)