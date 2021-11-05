import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


# 基础模块
class Block(nn.Module):
    def __init__(self):
        super(Block,self).__init__()
        self.conv1 = nn.Conv2d(3,3,1)
        self.conv2 = nn.Conv2d(3,3,1)
        self.conv3 = nn.Conv2d(3,3,1)
        self.conv4 = nn.Conv2d(3,3,1)
        self.conv5 = nn.Conv2d(3,3,1)
    def forward(self,x,res=None):
    # res存储每一层的输出信息，将它们包装威一个列表，并向后传递
        if res is None:
            res = []
        out0 = self.conv1(x)
        out0 += np.sum(res[:-1])
        res.append(out0)

        out1 = self.conv2(out0)
        out1 += np.sum(res[:-1])
        res.append(out1)


        out2 = self.conv3(out1)
        out2 += np.sum(res[:-1])
        res.append(out2)


        out3 = self.conv4(out2)
        out2 += np.sum(res[:-1])
        res.append(out2)

        out4 = self.conv5(out3)
        out2 += np.sum(res[:-1])
        res.append(out2)
            
        return out4,res
    
class Chain(nn.Sequential):
    def __init__(self,*args):
        super().__init__(*args)
        
# 下面这段注释代码功能上等同于super().__init__(*args),
# torch.nn.Sequential构造函数将传入的模块按照序号编好，然后放入__dict__中的_modules属性中
# 打印类对象时会取这个_modules属性
#         od = OrderedDict()
#         for i,arg in enumerate(args):
#             od["block"+str(i)] = arg
#         self.__dict__['_modules'] = od

    def forward(self,x):
        for idx,model in enumerate(self):
            if idx==0: 
                x,res = model(x)
            else:
                x,res = model(x,res)
        return x
            
        
        
## testcode
#model = Chain(Block(),Block())
#x = torch.ones((1,3,3,3))
#y_pred = model(x)

