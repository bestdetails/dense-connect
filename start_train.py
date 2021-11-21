# 开始训练
from my_model import Chain
from data_process import Dataset
import torch
import torch.nn as nn
import os

# def get_model():
#     model = Chain()
#     opt = torch.optim.SGD(model.parameters(),lr=1e-3)
#     return model,opt
# model,opt = get_model
# loss_func = nn.functional.cross_entropy
class Training_model():
  def __init__(self,
              model=Chain(num_class=10),
              loss_func=nn.functional.cross_entropy,
              opt=torch.optim.SGD,
              lr=1e-3,
              epochs=64,
              train_data=None,
              valid_data=None,
              test_data=None,
              batch_size=128): 
    self.model = model
    self.loss_func = loss_func
    self.lr = lr
    self.opt = opt(self.model.parameters(),lr=self.lr)
    self.epochs = epochs
    self.train_data = train_data
    self.valid_data = valid_data
    self.test_data = test_data
    self.batch_size = batch_size
    self.log_name = "./log/train_"+str(len(os.listdir("./log"))) + ".log"
    self.model_name = "./model/model_"+str(len(os.listdir("./model")))+".pt"
    os.system("touch "+self.log_name)
  def get_accuracy(self,y_pred,target):
    # target:one-hot vector
    return float((torch.argmax(y_pred,dim=1)==torch.argmax(target,dim=1)).float().mean())
  def train(self):
    if self.train_data is not None and self.valid_data is not None:
      
      for epoch in range(self.epochs):
        # because model exists batch normalization or dropout layer,
        # so we should set model.train() when training model
        # and set model.eval() when validating and testing model
        self.model.train()  
        f = open(self.log_name,'a')  # training log
        if epoch == 0:
          f.write("save model in "+self.model_name+"\n")
        for index,(epoch_data, epoch_label) in enumerate(self.train_data):
          y_pred = self.model(epoch_data)
          loss = self.loss_func(y_pred,epoch_label)
          acc = self.get_accuracy(y_pred,epoch_label)
          train_info = "Training epoch: "+str(epoch)+"/"+str(self.epochs)+"\t"+\
          str(index)+"/"+str(len(self.train_data))+\
          "\t" + "loss: "+str(float(loss))+"\t"+"acc: "+str(acc)+"\n"
          print(train_info[:-2])
          f.write(train_info)
          self.opt.zero_grad()
          loss.backward()
          self.opt.step()
        # validation
        self.model.eval()
        with torch.no_grad():
          losses = []
          accs = []
          for idx,(valid_data,valid_label) in enumerate(self.valid_data):
            pred = self.model(valid_data)
            validate_loss = self.loss_func(pred,valid_label)
            losses.append(validate_loss)
            validate_acc = self.get_accuracy(pred,valid_label)
            accs.append(validate_acc)
            validate_info = "Validating epoch: "+str(epoch)+"/"+str(self.epochs)+"\t"+\
            str(idx)+"/"+str(len(self.valid_data))+"\t"+"loss: "+\
            str(float(validate_loss))+"\t"+"acc: "+str(validate_acc)+"\n"
            f.write(validate_info)
            print(validate_info[:-2])
          l = sum(losses) / len(self.valid_data)
          a = sum(accs) / len(self.valid_data)
          info = "Epoch"+str(epoch)+" Validating Loss: "+\
          str(float(l))+"\t"+"Validating Acc: "+str(a)+"\n"
          f.write(info)
        f.close()
      
      torch.save(self.model,self.model_name)
    else:
      raise TypeError("Type of train_data or valid_data is None,expected torch.nn.DataLoader")

  def test(self):
    if self.test_data is not None:
      self.model.eval()
      with torch.no_grad():
        losses = []
        accs = []
        f = open(self.log_name,'a')
        for idx,(test_data,test_label) in enumerate(self.test_data):
          pred = self.model(test_data)
          test_loss = self.loss_func(pred,test_label)
          losses.append(test_loss)
          test_acc = self.get_accuracy(pred,test_label)
          accs.append(test_acc)
          test_info = str((idx+1)*self.batch_size)+"\t"+"loss: "+\
              str(float(test_loss))+"\t"+"acc: "+str(test_acc)+"\n"
          f.write(test_info)
        l = sum(losses) / len(self.test_data)
        a = sum(accs) / len(self.test_data)
        info = "Test Loss: "+str(float(l))+'\t'+"Test Acc: "+str(a)+"\n"
        f.write(info)
      f.close()
    else:
      raise TypeError("Type of test_data is None, expected torch.nn.DataLoader")

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class WrappedDataLoader:
  def __init__(self, dl, func):
    self.dl = dl
    self.func = func

  def __len__(self):
    return len(self.dl)

  def __iter__(self):
    batches = iter(self.dl)
    for b in batches:
      yield (self.func(*b))  
def process(x,y):
  return x.to(dev),y.to(dev)





# print(len(train),len(validation),len(test))        
model = Chain(num_class=10)
loss_func = nn.functional.cross_entropy
opt = torch.optim.SGD
batch_size=256
root = "/mnt/storage-data/zhengzepeng/cifar10/cifar-10-batches-py"
train_path = ("data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5")
test_path = ("test_batch",)
dataset = Dataset(root_path=root,batch_size=batch_size,shuffle=True,
is_validate=True,train_paths=train_path,test_paths=test_path)
train,validation = dataset.train_data()
test = dataset.test_data()

train = WrappedDataLoader(train,process)
validation = WrappedDataLoader(validation,process)
test = WrappedDataLoader(test,process)
model.to(dev)

training_process = Training_model(model=model,loss_func=loss_func,
opt=opt,lr=1e-3,epochs=200,train_data=train,valid_data=validation,test_data=test,
batch_size=batch_size)

training_process.train()
