from utils import *

import torch.nn as nn
import torch.nn.functional as F

# help: https://www.youtube.com/watch?v=o_3mboe1jYI
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        # input = in_channel, output = in_channel (kernel 3, paddin 1, stride 1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=3, stride=stride,padding=1)  
        # Use batchNorm to re-normalize features
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Pass through ReLu
        self.relu = nn.ReLU(inplace=True)
        # # input = out_channel, output = out_channel (kernel 3, paddin 1, stride 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)  
        # Use batchNorm to re-normalize features
        self.bn2 = nn.BatchNorm2d(out_channels) 
        self.downsample = downsample

    def forward(self, x):
        # save x to use as identity function later
        identity = x

        # pass x through conv layers
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # downsample if required
        if self.downsample is not None:
            identity = self.downsample(x)

        # add saved identity features to our current set of features in the network
        # done element wise
        out += identity
        # pass thought activation function
        out = self.relu(out)
        # return as output of our residual block
        return out