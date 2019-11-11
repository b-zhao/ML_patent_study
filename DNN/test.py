import torch

from torch.utils import data as Data
import torch.backends.cudnn as cudnn
import matplotlib

import numpy as np
matplotlib.use('agg')
from tqdm import tqdm

import datetime
from loader import  PatentDataset, load_data_test, load_data_with_convert_Y
from DNN.model import Net, Net_with_softmax


def test(opt):

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

    xtest, ytest = load_data_test()

    datasets = {}
    datasets['test'] = PatentDataset(xtest, ytest)

    N_input = xtest.shape[1]
    N_output = ytest.shape[1]

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=opt.batchsize,
                                                  shuffle=True, num_workers=8, pin_memory=True)
                   for x in ['test']}

    use_gpu = torch.cuda.is_available()

    model =  Net_with_softmax(N_input, 1024, N_output) if opt.convert_y else Net(N_input, 1024, N_output)

    model.load_state_dict(torch.load(opt.model_dir))

    # if opt.model_dir and opt.train_on_save:
    #     model.load_state_dict(torch.load(opt.model_dir))

    model = model.double().cuda()


    loss_function = torch.nn.MSELoss()


    def exec(model, criterion, num_epochs=25):
        model.train(False)

        error = 0.0
        running_loss = 0.0
        true_predict = 0
        totalNum = 0

        pbar = tqdm(dataloaders['test'])
        for inputs, labels in pbar:
            batch_size, N = inputs.shape
            totalNum += batch_size

            if use_gpu:
                inputs = inputs.double().cuda()
                labels = labels.double().cuda()

            outputs = model(inputs)

            predict_result = outputs.cpu().detach().numpy().reshape(-1)
            ground_truth = labels.cpu().detach().numpy().reshape(-1)

            true_predict += np.where( abs(predict_result - ground_truth) < 180)[0].shape[0]
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            error += np.sum(abs(predict_result - ground_truth))


        print('loss: ', running_loss / totalNum)
        print('average error: ', error / totalNum)
        print('accuracy: ', true_predict / totalNum)



        # do final operation here
    exec(model, loss_function, num_epochs=opt.epoch)