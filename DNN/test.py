import torch

from torch.utils import data as Data
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

import datetime
from loader import  PatentDataset, load_data_test, load_data_with_convert_Y
from DNN.model import Net, Net_with_softmax


def plot_sample_figure(predict, ground_truth):
    plt.title('Loss')
    x = np.arange(25)

    plt.plot(x, predict[:25], color='green', label='prediction')
    plt.plot(x, ground_truth[:25], color='red', label='ground truth')


    # plt.plot(x_axix, train_pn_dis,  color='skyblue', label='PN distance')
    # plt.plot(x_axix, thresholds, color='blue', label='threshold')
    plt.legend()  # 显示图例

    plt.xlabel('samples')
    plt.ylabel('granted days')
    plt.savefig('./sample.jpg')



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

    xtest, ytest = load_data_test(opt.with_if)

    datasets = {}
    datasets['test'] = PatentDataset(xtest, ytest)

    N_input = xtest.shape[1]
    N_output = ytest.shape[1]
    size = xtest.shape[0]


    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=opt.batchsize,
                                                  shuffle=True, num_workers=8, pin_memory=True)
                   for x in ['test']}

    use_gpu = torch.cuda.is_available()

    model =  Net_with_softmax(N_input, 512, N_output) if opt.convert_y else Net(N_input, 512, N_output)

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

        all_predict = np.zeros(size)
        all_gt = np.zeros(size)

        for inputs, labels in pbar:
            batch_size, N = inputs.shape
            totalNum += batch_size

            if use_gpu:
                inputs = inputs.double().cuda()
                labels = labels.double().cuda()

            outputs = model(inputs)


            predict_result = outputs.cpu().detach().numpy().reshape(-1)
            ground_truth = labels.cpu().detach().numpy().reshape(-1)

            all_predict[totalNum - batch_size: totalNum] = predict_result
            all_gt[totalNum - batch_size: totalNum] = ground_truth


            true_predict += np.where( abs(predict_result - ground_truth) < 180)[0].shape[0]
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            error += np.sum(abs(predict_result - ground_truth))

        print('loss: ', running_loss / totalNum)
        print('average error: ', error / totalNum)
        print('accuracy: ', true_predict / totalNum)
        plot_sample_figure(all_predict, all_gt)



        # do final operation here
    exec(model, loss_function, num_epochs=opt.epoch)