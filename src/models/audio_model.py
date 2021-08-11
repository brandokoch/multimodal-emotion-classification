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
        h= torch.flatten(x, start_dim=1)
        h = self.fc1(h)
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

class BaselineAudioModel_2(nn.Module):
    def __init__(self):
        super(BaselineAudioModel_2, self).__init__()

        def conv(ni, nf, ks=3, act=True):
            res=nn.Conv2d(ni,nf, stride=2, kernel_size=ks, padding=ks//2)
            if act: res = nn.Sequential(res, nn.ReLU())
            return res

        self.conv1 = conv(1,4)
        self.conv2= conv(4,8)
        self.conv3= conv(8,16)
        self.conv4= conv(16,32)
        self.conv5= conv(32,64)
        self.conv6=conv(64, 128, act=False)
        self.flatten=nn.Flatten()
        self.lin1= nn.Linear(1024, 512)
        self.lin2= nn.Linear(512, 64)
        self.lin3= nn.Linear(64, 3)

    def forward(self, x):
        h= self.conv1(x)
        h= self.conv2(h)
        h= self.conv3(h)
        h= self.conv4(h)
        h= self.conv5(h)
        h= self.conv6(h)
        h = self.flatten(h)
        h= self.lin1(h)
        h= self.lin2(h)
        h= self.lin3(h)
        return h

class BaselineAudioModel_3(nn.Module):
    def __init__(self,  n_input=1, n_output=3, stride=16, n_channel=32):
        super().__init__()

        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

        self.flatten=nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x= self.flatten(x)
        x = self.fc1(x)

        # We dont apply softmax to output since nn.CrossEntropyLoss 
        # combines LogSoftmax and NLLLoss

        return x

if __name__=='__main__':
    print('UNIT TEST AudioBaselineModel_3:')
    x=torch.zeros(32, 1, 57515)
    model = BaselineAudioModel_3()
    out=model(x)
    print('\t out shape: ', out.size())
    print('AudioBaselineModel_3 test PASSED')

