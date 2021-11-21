# cifar10原始数据处理
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
from random import shuffle
root = "/mnt/storage-data/zhengzepeng/cifar10/cifar-10-batches-py"
class Dataset():
    def __init__(self,root_path=root,batch_size=1,shuffle=False,is_validate=False,split=0.1,num_works=8,**kwargs):
        self.batch_size = batch_size
        self.root_path = root_path
        self.shuffle=shuffle
        self.is_validate=is_validate
        self.split = split if self.is_validate is True else None
        self.num_works = num_works
        if kwargs is not None:
            if "train_paths" in kwargs.keys():
                type_arg = type(kwargs['train_paths'])
                if type_arg is not list and type_arg is not tuple:
                    raise TypeError("type of train_paths should be list or tuple, got {}".format(type_arg))
                
                self.train_paths = []
                for path in kwargs['train_paths']:
                    self.train_paths.append(os.path.join(self.root_path,path))
                    
            if "test_paths" in kwargs.keys():
                type_arg = type(kwargs['test_paths'])
                if type_arg is not list and type_arg is not tuple:
                    raise TypeError("type of test_paths should be list or tuple, got {}".format(type_arg))
                self.test_paths = []
                for path in kwargs['test_paths']:
                    self.test_paths.append(os.path.join(self.root_path, path))

    def unpickle(self,file_path):
        with open(file_path, 'rb') as fo:
            imgdict = pickle.load(fo, encoding='bytes')
        return imgdict
    
    def get_sample(self,data):
        new_data = []
        for sample in data:
            red = sample[0:1024].reshape((32,32))
            green = sample[1024:2048].reshape((32,32))
            blue = sample[2048:].reshape((32,32))
            new_sample = np.stack((red,green,blue))
            new_data.append(new_sample)
        return np.asarray(new_data)  # (10000,3,32,32)
        
    
    def train_data(self):
        if self.train_paths is not None:
            img_list = []
            label_list = []
            for path in self.train_paths:
                file = self.unpickle(path)
                data = self.get_sample(file[b'data'])# (10000,3,32,32)
                label = self.get_label(file[b'labels'])
                img_list.append(data)
                label_list.append(label)
            #print(len(img_list))
            #random_index = [i for i in range(len(img_list))]
            img = img_list[0]
            label = label_list[0]
            for il in img_list[1:]:
                img = np.concatenate((img, il))
            for ll in label_list[1:]:
                label = np.concatenate((label,ll))
            img = torch.tensor(img,dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)

            if self.is_validate:
                random_index = [i for i in range(len(img))]
                shuffle(random_index)
                with open("index",'w') as f:
                    f.write(str(random_index))
                bound_index = int(len(img)*(1-self.split))
                train_img = img[random_index[0:bound_index]]
                train_label = label[random_index[0:bound_index]]
                validate_img = img[random_index[bound_index:]]
                validate_label = label[random_index[bound_index:]]
                train_dataset = TensorDataset(train_img, train_label)
                validate_dataset = TensorDataset(validate_img,validate_label)
                return (DataLoader(train_dataset,batch_size=self.batch_size,
                shuffle=self.shuffle,num_workers=self.num_works),
                DataLoader(validate_dataset,batch_size=self.batch_size,
                shuffle=False,num_workers=self.num_works))
            else:
                dataset = TensorDataset(img,label)
                return DataLoader(dataset, batch_size=self.batch_size,
                shuffle=self.shuffle,num_workers=self.num_works)
        return None
    def test_data(self):
        if self.test_paths is not None:
            img_list = []
            label_list = []
            for path in self.test_paths:
                file = self.unpickle(path)
                data = self.get_sample(file[b'data'])# (10000,3,32,32)
                label = self.get_label(file[b'labels'])
                img_list.append(data)
                label_list.append(label)
            img = img_list[0]
            label = label_list[0]
            for il in img_list[1:]:
                img = np.concatenate((img, il))
            for ll in label_list[1:]:
                label = np.concatenate((label,ll))
            img = torch.tensor(img,dtype=torch.float32)
            label = torch.tensor(label,dtype=torch.float32)
            
            dataset = TensorDataset(img,label)
            return DataLoader(dataset, batch_size=self.batch_size,
            shuffle=self.shuffle,num_workers=self.num_works)
        return None
            
    def get_label(self,labels):
        l = []
        for label in labels:
            l.append(np.eye(10)[label])
        return np.asarray(l)
    def show_img(img):
        plt.imshow(img.transpose((1,2,0)))


