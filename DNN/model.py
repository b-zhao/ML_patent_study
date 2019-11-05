
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()

        self.hidden = nn.Linear(n_feature, n_hidden)
        self.sigmoid = torch.sigmoid
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.predict(x)
        return x
