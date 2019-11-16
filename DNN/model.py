
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()

        self.hidden = nn.Linear(n_feature, n_output)
        #self.sigmoid = torch.relu
        #self.predict1 = nn.Linear(n_hidden, n_hidden)
        #self.predict2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden(x)
        #x = self.sigmoid(x)
        #x = self.predict1(x)
        #x = self.sigmoid(x)
        #x=  self.predict2(x)
        return x


class Net_with_softmax(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net_with_softmax, self).__init__()

        self.hidden = nn.Linear(n_feature, n_hidden)
        self.relu = torch.relu
        self.predict1 = nn.Linear(n_hidden, n_hidden)
        self.predict2 = nn.Linear(n_hidden, n_output)

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.predict1(x)
        x = self.relu(x)
        x = self.predict2(x)
        x = self.softmax(x)
        return x
