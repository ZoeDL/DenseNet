import torch
import torch.nn as nn

'''
import torch.autograd as Variable
import torch.utils.data as Data

import torchvision.datasets as DataSet
import torchvision.transforms as Transforms

import torch.optim as Optim
'''
import torch.nn.functional as F

import math


class Bottleneck(nn.Module):
    def __init__(self,inChannels,growthRate):
        super(Bottleneck,self).__init__()
        innerChannels=4*growthRate
        self.bn1=nn.BatchNorm2d(inChannels)
        self.conv1=nn.Conv2d(in_channels=inChannels,out_channels=innerChannels,kernel_size=1,bias=False)
        self.bn2=nn.BatchNorm2d(innerChannels)
        self.conv2=nn.Conv2d(in_channels=innerChannels,out_channels=growthRate,kernel_size=3,padding=1,bias=False)

    def forward(self,x):
        out=self.conv1(F.relu(self.bn1(x)))
        out=self.conv2(F.relu(self.bn2(out)))
        out=torch.cat((x,out),dim=1)      # concat over depth-wise
        return out


class Transition(nn.Module):
    def __init__(self,inChannels, innerChannels):
        super(Transition,self).__init__()
        self.bn=nn.BatchNorm2d(inChannels)
        self.conv=nn.Conv2d(inChannels,innerChannels,1,bias=False)

    def forward(self,x):
        out=self.conv(F.relu(self.bn(x)))
        out=F.avg_pool2d(out,2,2)
        return out


class Densenet(nn.Module):
    def __init__(self,growthRate,reduction,nLayers,nClasses):
        super(Densenet,self).__init__()
        inChannels=int(2*growthRate)
        self.conv1=nn.Conv2d(1,inChannels,3,1,1)
        self.dense1=self.make_denseBlock(inChannels,nLayers,growthRate)
        inChannels+=nLayers*growthRate
        outchannels=int(math.floor(reduction*inChannels))
        self.transition1=Transition(inChannels,outchannels)
        inChannels=outchannels
        self.bn1=nn.BatchNorm2d(inChannels)
        self.fc = nn.Linear(inChannels,nClasses)

    def make_denseBlock(self,inChannels,nLayers,growthRate):
        layers=[]
        for i in range(nLayers):
            layers.append(Bottleneck(inChannels,growthRate))
            inChannels+=growthRate
        return nn.Sequential(*layers)

    def forward(self,x):
        out=F.avg_pool2d(self.conv1(x),2,2)
        out=self.dense1(out)
        out=self.transition1(out)
        # squeeze out all the dimensions of size 1 and return a tensor
        #out=torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)),7))
        out=F.avg_pool2d(F.relu(self.bn1(out)),7)
        out = out.view(out.size(0),-1)
        out=self.fc(out)
        return out

