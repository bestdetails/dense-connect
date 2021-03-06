#算是最终版本

import torch
import torchvision
import numpy as np
import torch.nn as nn
import math


class BasicBlock(nn.Module):
    def __init__(self,in_channels=1,out_channels=1,stride=1,downsamples_1=None,downsamples_2=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                               kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                               kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = (stride,1)
        self.downsamples_1 = downsamples_1
        self.downsamples_2 = downsamples_2
        self.in_channels = (in_channels,out_channels)
        self.out_channels = (out_channels,out_channels)

    def forward(self,x,samples=None):
        if samples is None:
            samples = []
            
        #print("x:",x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        # 加到bn层后面
        if len(self.downsamples_1) > 0:
            for idx,sample in enumerate(samples[0:-1]):
                
                if self.downsamples_1[idx] is not None:
                    #print("out:",out.shape," self.downsamples_1[idx](sample):",self.downsamples_1[idx](sample).shape)
                    out += self.downsamples_1[idx](sample)
                    #out += 0.1
                else:
                    out += sample
        out = self.relu(out)
        #print("out1:",out.shape)
        samples.append(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        if len(self.downsamples_2) > 0:
            for idx,sample in enumerate(samples[0:-1]):
                if self.downsamples_2[idx] is not None:
                    out += self.downsamples_2[idx](sample)
                else:
                    out += sample
        
        out = self.relu(out)
        #print("out2:",out.shape)
        samples.append(out)
        
        return out,samples
    
class Chain(nn.Module):
    def __init__(self,*args):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(7,7),padding=(3,3),stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        layers = []
        layer1 = self.make_layer(in_channels=64,out_channels=64,n=2,is_downsample=False,stride=1)
        blocks = self.add_downsamples(layers,layer1)
        #print("blocks = ",layer1)
        self.layer1 = Denseq(*blocks)
        for l in layer1:
            layers.append(l)

        layer2 = self.make_layer(in_channels=64,out_channels=128,n=2)
        blocks = self.add_downsamples(layers,layer2)
        self.layer2 = Denseq(*blocks)
        for l in layer2:
            layers.append(l)

        
        layer3 = self.make_layer(in_channels=128,out_channels=256,n=2)
        blocks = self.add_downsamples(layers,layer3)
        self.layer3 = Denseq(*blocks)
        for l in layer3:
            layers.append(l)

        
        layer4 = self.make_layer(in_channels=256,out_channels=512,n=2)
        blocks = self.add_downsamples(layers,layer4)
        self.layer4 = Denseq(*blocks)
        for l in layer4:
            layers.append(l)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def add_downsamples(self,ls,layer:list):
        # layers是多个层的所有block列表，layer是当前层的block列表
        ds = []
        downsamples_1 = []
        downsamples_2 = []
        new_layer = []
        layers = copy.deepcopy(ls)
        for i,b in enumerate(layer):
            current_inchannels = b.in_channels[0]
            current_outchannels = b.out_channels[0]
            current_stride = b.stride[0]
            # block里面的两个conv和网络最开始的conv1做适配
            downsample_pool_1 = self.get_downsamples(b,self.conv1,(b.in_channels[0],b.out_channels[0]),
                                                 (self.conv1.in_channels,self.conv1.out_channels))
            downsample_pool_2 = self.get_downsamples(b,self.conv1,(b.in_channels[1],b.out_channels[1]),
                                                 (self.conv1.in_channels,self.conv1.out_channels))
            downsamples_1.append(downsample_pool_1)
            downsamples_2.append(downsample_pool_2)
            for j,block in enumerate(layers):
                # block里面的两个conv和其他block的两个conv做适配
                downsample_1_1 = self.get_downsamples(b,block,(b.in_channels[0],b.out_channels[0]),
                                                 (block.in_channels[0],block.out_channels[0]))
                downsample_1_2 = self.get_downsamples(b,block,(b.in_channels[0],b.out_channels[0]),
                                                 (block.in_channels[1],block.out_channels[1]))
                downsamples_1.append(downsample_1_1)
                downsamples_1.append(downsample_1_2)
                
                downsample_2_1 = self.get_downsamples(b,block,(b.in_channels[1],b.out_channels[1]),
                                                      (block.in_channels[0],block.out_channels[0]))
                downsample_2_2 = self.get_downsamples(b,block,(b.in_channels[1],b.out_channels[1]),
                                                      (block.in_channels[1],block.out_channels[1]))
                downsamples_2.append(downsample_2_1)
                downsamples_2.append(downsample_2_2)
            # 对于downsample_1，最后一个downsample不需要
            if len(downsamples_1)>0:
                downsamples_1.pop(-1)
            # 适配完了就添加到列表中，组装成新的layer
            # 重新创建BasicBlock对象
            new_basicblock = BasicBlock(current_inchannels,current_outchannels,
                                        current_stride,nn.Sequential(*downsamples_1),nn.Sequential(*downsamples_2))
            # 添加到列表中
            new_layer.append(new_basicblock)
            downsamples_1 = []
            downsamples_2 = []
            layers.append(b)
        return new_layer
        
    def make_layer(self,in_channels:int,out_channels:int,n:int,is_downsample:bool=True,stride:int=2):
        layer = []
        if is_downsample:
            seq = BasicBlock(in_channels,out_channels,stride)
        else:
            seq = BasicBlock(out_channels,out_channels,1)
        layer.append(seq)
        for _ in range(n-1):
            layer.append(BasicBlock(out_channels,out_channels,1))
            
        return layer
        
    def get_downsamples(self,m,n,m_channels,n_channels): # m在后，n在前
        # 仅仅做两个层之间的形状适配
        
        extension = int(math.log(m_channels[1]//n_channels[1],2))
        seq = []
        if extension != 0:
            for i in range(extension):
                seq.append(nn.Conv2d(n_channels[1]*2**i,n_channels[1]*2**(i+1),kernel_size=1,stride=2))
                seq.append(nn.BatchNorm2d(n_channels[1]*2**(i+1)))
            return nn.Sequential(*seq)
        else:
            return None
        
        
    def forward(self,x):
        samples = []
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        samples.append(out)
        
        out,samples = self.layer1(out,samples)
        
        out,samples = self.layer2(out,samples)
        
        out,samples = self.layer3(out,samples)
        out,samples = self.layer4(out,samples)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out
    
# block = Block()
t1 = torch.rand((10,3,224,224))
# #block(t1)
# # block(t1).shape
class Denseq(nn.Sequential):
    def __init__(self,*args):
        super().__init__(*args)
        
    def forward(self,x,samples=None):
        
        for i,block in enumerate(self):
            x,samples = block(x,samples)
        return x,samples



chain = Chain()
#chain.down_samples['downsample_0']
chain(t1)
