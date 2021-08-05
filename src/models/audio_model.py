import torch
from torch import nn
import torch.nn.functional as F


class BaselineAudioModel_1(nn.Module):
    def __init__(self, input_size):
        super(BaselineAudioModel_1, self).__init__()
        self.input_size = input_size
        
        self.fc1 = nn.Linear(6280, 2000)
        self.bc1 = nn.BatchNorm1d(2000)
        
        self.fc2 = nn.Linear(2000, 200)
        self.bc2 = nn.BatchNorm1d(200)
        
        self.fc3 = nn.Linear(200, 100)
        self.bc3= nn.BatchNorm1d(100)

        self.fc4= nn.Linear(100, 7)

        
        
    def forward(self, x):
        # flatten
        h = self.fc1(x)
        h = self.bc1(h)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        
        h = self.fc2(h)
        h = self.bc2(h)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)

        h=self.fc3(h)
        h= self.bc3(h)
        h=torch.relu(h)
        h=F.dropout(h, p=0.5, training=self.training)
        
        out = self.fc4(h)

        # We dont apply softmax to output since nn.CrossEntropyLoss 
        # combines LogSoftmax and NLLLoss

        return out
