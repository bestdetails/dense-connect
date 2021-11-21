import torch
import math
import torch.nn as nn
import copy

class BasicBlock(nn.Module):
    def __init__(self,in_channels=1,out_channels=1,stride=1,downsamples_1=None,downsamples_2=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = (stride,1)
        self.downsamples_1 = downsamples_1
        self.downsamples_2 = downsamples_2
        self.in_channels = (in_channels,out_channels)
        self.out_channels = (out_channels,out_channels)

#         if downsamples is not None:
#             #print(type(self.downsamples))
#             for i,ds in enumerate(downsamples):
#                     self._modules["downsample"+str(i)] = ds
    def forward(self,x,res=None):
        identity = x
        if res is None:
            res = []
            
        
        out = self.conv1(x)
        out = self.bn1(out)
        for i,downsample in enumerate(self.downsamples):
            if downsample is not None:
                out += downsample(res[i])
            else:
                out += res[i]
        out = self.relu(out)
        res.append(out)
        
            
        return out,res
    
class Chain(nn.Module):
    def __init__(self,*args):
        super().__init__()
#         layers = []
#         for arg in args:
#             layers.append(arg) 
        self.down_samples = {}
        
        
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(7,7),padding=(3,3),stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        layers = []
        layer1 = self.make_layer(in_channels=64,out_channels=64,n=2,is_downsample=False,stride=1)
        blocks = self.add_downsamples(layers,layer1)
        #print("blocks = ",layer1)
        self.layer1 = nn.Sequential(*blocks)
        for l in layer1:
            layers.append(l)

        layer2 = self.make_layer(in_channels=128,out_channels=128,n=2)
        blocks = self.add_downsamples(layers,layer2)
        self.layer2 = nn.Sequential(*blocks)
        for l in layer2:
            layers.append(l)

        
        layer3 = self.make_layer(in_channels=256,out_channels=256,n=2)
        blocks = self.add_downsamples(layers,layer3)
        self.layer3 = nn.Sequential(*blocks)
        for l in layer3:
            layers.append(l)

        
        layer4 = self.make_layer(in_channels=512,out_channels=512,n=2)
        blocks = self.add_downsamples(layers,layer4)
        self.layer4 = nn.Sequential(*blocks)
        for l in layer4:
            layers.append(l)

        
#         for layer in [layer1,layer2,layer3,layer4]:
#             for l in layer:
#                 layers.append(l)
        
        

        #self.layer = nn.Sequential(*blocks)
            
            # downsample_x 为Sequential()，里面没有东西时，它的len为0，意味着不需要任何连接
            # 如果里面有东西但是为None，意味着直接连接，
            # 如果有东西且不为None，说明需要经过1X1卷积以适配形状再连接
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
            # 适配完了就添加到列表中，准备组装成新的layer
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
            seq = BasicBlock(in_channels,out_channels,1)
        layer.append(seq)
        for _ in range(n-1):
            layer.append(BasicBlock(in_channels,out_channels,1))
            
        return layer
        
    def get_downsamples(self,m,n,m_channels,n_channels): # m在后，n在前
        # 仅仅做两个层之间的形状适配
#         m_channels = (m.in_channels,m.out_channels)
#         n_channels = (n.in_channels,n.out_channels)
        
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
        for idx,model in enumerate(self):
            if idx==0:
                x,res = model(x)
            else:
                x,res = model(x,res)
        return x
            
        
        
# block = Block()
# t1 = torch.rand((10,3,224,224))
# #block(t1)
# # block(t1).shape
chain = Chain()
#chain.down_samples['downsample_0']
chain
