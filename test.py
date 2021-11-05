import torch
import torch.nn as nn

class Test(nn.Module):
    def __init__(self,in_feature=1,out_feature=1,downsamples=None):
        super().__init__()
        self.fc = nn.Linear(10,10,bias=False)
        # 如果将nn.Module的子类对象初始化赋值，会将这个属性放入__dict__的_modules中
        # 它们内部的参数也会被自动捕捉
        
        # 如果downsamples是一个nn.modules类对象，那么就会自动加入_modules中
        # 但如果是其他的东西，那么只会加入__dict__，跟普通的类属性一样，而不是加入到_modules
        self.downsamples=downsamples
        
        # nn.Parmeter只接受tensor类对象
        # 用nn.Parameter将其他类型注册为模型的参数，这些参数也就跟其他可训练参数一样了
        self.t = nn.Parameter(torch.tensor([1,2],dtype=torch.float32))
        
#测试
samples = [nn.Sequential(nn.Linear(10,10,bias=True),nn.Linear(10,10,bias=True)),
           nn.Sequential(nn.Linear(10,10,bias=True),nn.Linear(10,10,bias=True))]
test = Test(downsamples=samples)

# 在这里downsamples仅仅注册为一个普通的属性，而fc则注册进了_modules
# 可以查看
# <<< test.__dict__
# {'training': True,
#  '_parameters': OrderedDict([('t', Parameter containing:
#                tensor([1., 2.], requires_grad=True))]),

#  ...

#  '_modules': OrderedDict([('fc',
#                Linear(in_features=10, out_features=10, bias=False))]),
#  'downsamples': [Sequential(
#     (0): Linear(in_features=2, out_features=2, bias=True)
#     (1): Linear(in_features=10, out_features=10, bias=True)
#   ),
#   Sequential(
#     (0): Linear(in_features=10, out_features=10, bias=True)
#     (1): Linear(in_features=10, out_features=10, bias=True)
#   )]}

# <<<test._modules
#OrderedDict([('fc', Linear(in_features=10, out_features=10, bias=False))])

# 这里注册为参数的t可以在_parameters中看到

# <<<test._parameters
# OrderedDict([('t',
#               Parameter containing:
#               tensor([1., 2.], requires_grad=True))])

# test.parameters()是一个生成器，可以遍历它看到所有可训练的参数
for p in test.parameters():
  print(p)
  
Parameter containing:
tensor([1., 2.], requires_grad=True) # 这个是用nn.Parameters()注册的t
Parameter containing:
tensor([...],  # 这个tensor是fc层的参数，形状为(10,10)
       requires_grad=True)
# 经过nn.Parameters()注册过的参数和nn.Module类对象作为属性的都是可训练的参数
# 而其他的属性downsamples，它是nn.Module类对象的列表，但被注册成了普通属性。


# 因此应该遍历downsamples将其注册进来，成为_modules的成分，下面是一个实现
class Test(nn.Module):
    def __init__(self,in_feature=1,out_feature=1,downsamples=None):
        super().__init__()
        self.fc = nn.Linear(10,10,bias=False)
        self.downsamples=downsamples
        for i,ds in enumerate(self.downsamples):
            self._modules['downsample'+str(i)] = ds
        self.t = nn.Parameter(torch.tensor([1,2],dtype=torch.float32))
samples = [nn.Sequential(nn.Linear(10,10,bias=True),nn.Linear(10,10,bias=True)),
           nn.Sequential(nn.Linear(10,10,bias=True),nn.Linear(10,10,bias=True))]
test  = Test(downsamples=samples)
# 这次循环后可以在_modules中看到，也可以在parameters()中看到
#<<<test._modules
# OrderedDict([('fc', Linear(in_features=10, out_features=10, bias=False)),
#              ('downsample0',
#               Sequential(
#                 (0): Linear(in_features=10, out_features=10, bias=True)
#                 (1): Linear(in_features=10, out_features=10, bias=True)
#               )),
#              ('downsample1',
#               Sequential(
#                 (0): Linear(in_features=10, out_features=10, bias=True)
#                 (1): Linear(in_features=10, out_features=10, bias=True)
#               ))])

for p in test.parameters():
    print(p.shape)
# torch.Size([2])
# torch.Size([10, 10])
# torch.Size([10, 10])
# torch.Size([10])
# torch.Size([10, 10])
# torch.Size([10])
# torch.Size([10, 10])
# torch.Size([10])
# torch.Size([10, 10])
# torch.Size([10])
    
  
# 在构造函数中只是将所有网络成分初始化好，成分之间的联系尚未确定，而这些联系，比如resnet中的残差连接
# 需要在forward方法中构建，数据传递调用进来的时候经过forward构建的联系组成resnet
  
#
        
