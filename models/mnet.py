import torch
import torch.nn as nn
import torch.nn.functional as F


class MNET(nn.Module):

    def __init__(self):
        super(MNET, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2= nn.Conv2d(10,20,kernel_size=5)
        self.conv_dropout = nn.Dropout2d()
        self.fc = nn.Linear(320,128)
        #self.fc2 = nn.Linear(60,10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv_dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1,320)
        x = F.relu(self.fc(x))
        
        return x

def mnist_model():
    return {'backbone': MNET(), 'dim': 128}