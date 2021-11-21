# 新的想法实现
# BasicBlock初始不带任何downsample，传递进Chain后，由Chain的构造函数
# 负责获取各个BasicBlock的状态，为每个BasicBlock拿到其对应的downsample
# 接着重新创建BasicBlock对象，把获取到的downsample传递进BasicBlock构造函数
# 这个过程相当于创建两次BasicBlock对象，但是不会像test2.py那样，每个block
# 对应的downsample都与该block分开（由此需要特别标注downsample的名字），
# 当前实现可以之间在block中访问到对应的downsample
import torch
import torch.nn as nn
import math
class BasicBlock(nn.Module):
    def __init__(self,in_channels=1,out_channels=1,stride=1,downsamples=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsamples = downsamples
        self.in_channels = in_channels
        self.out_channels = out_channels

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
        self.down_samples = {}
        ds = []
        blocks = []
        for i,module in enumerate(layers):
            in_channels = module.in_channels
            out_channels = module.out_channels
            stride = module.stride
            
            for j,m in enumerate(layers[0:i]):
                down_sample = self.get_downsamples(module,m)
                ds.append(down_sample)
                
            self.down_samples['downsample_'+str(i)] = nn.Sequential(*ds)  # 无实际意义，但是取downsample会更方便
            # 重新创建对象
            basicblock = BasicBlock(in_channels,out_channels,stride,nn.Sequential(*ds))
            blocks.append(basicblock)
            #self._modules['downsample_'+str(i)] = nn.Sequential(*ds)
            ds = []
        self.layer = nn.Sequential(*blocks)
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
# 测试
chain = Chain(BasicBlock(64,64),BasicBlock(64,64),BasicBlock(64,128),BasicBlock(128,128),BasicBlock(128,256))
print(chain)

# Chain(
#   (layer): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsamples): Sequential()
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsamples): Sequential(
#         (0): None
#       )
#     )
#     (2): BasicBlock(
#       (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsamples): Sequential(
#         (0): Sequential(
#           (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2))
#           (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#         (1): Sequential(
#           (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2))
#           (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#     )
#   ...
#   ...
#
#


