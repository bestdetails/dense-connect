# 密集连接通用化版本1，仍然是测试Block，这里的Block只有一层CNN
# 主要在Chain中添加为每个Block获取前置所有输入的函数get_downsample
import torch
import torch.nn as nn
import math

class BasicBlock(nn.Module):
    def __init__(self,in_feature=1,out_feature=1,stride=1,downsamples=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_feature,out_channels=out_feature,kernel_size=3,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        # 可认为downsamples在此处基本没有作用
        self.downsamples = downsamples
        self.in_channels = in_feature
        self.out_channels = out_feature

        if self.downsamples is not None:
            if type(self.downsamples) is list or type(self.downsamples) is tuple:
                for i,ds in enumerate(self.downsamples):
                    self._modules["downsample"+str(i)] = ds
            else:
                pass
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
        layers = []
        for arg in args:
            layers.append(arg)
        self.layer = nn.Sequential(*layers)

        self.down_samples = {}
        ds = []
        
        for i,module in enumerate(layers):
            for j,m in enumerate(layers[0:i]):
                down_sample = self.get_downsamples(module,m)
                
                ds.append(down_sample)

            self.down_samples['downsample_'+str(i)] = nn.Sequential(*ds)
            self._modules['downsample_'+str(i)] = nn.Sequential(*ds)
            ds = []
    def get_downsamples(self,m,n): # m在后，n在前
        m_channels = (m.in_channels,m.out_channels)
        n_channels = (n.in_channels,n.out_channels)
        
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
      
      
      
      
      
  
  
  
  
