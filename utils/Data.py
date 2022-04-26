# !/usr/bin/python
# -*- coding:utf-8 -*-

import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
class FV(Dataset):

    def __init__(self, root,  train=True, transform=None,target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.label = []
        self.data = []

        if self.train:
            file_list_label = os.listdir(self.root + '/train')
            file_list_label.sort(key=lambda x: int(x[:])) 
            for index, i in enumerate(file_list_label):
                file_list_img = os.listdir( self.root + '/train/' + i)  
                for j in file_list_img:
                    imge = Image.open(root + '/train/' + i + '/' + j).convert('L')
                    imge = imge.resize((200, 91))

                    imge = np.array(imge)

                    self.label.append(index)
                    self.data.append(imge)
        else:
            file_list_label = os.listdir(self.root + '/test')
            file_list_label.sort(key=lambda x: int(x[:]))  
            for index, i in enumerate(file_list_label):
                file_list_img = os.listdir(self.root + '/test/' + i) 
                for j in file_list_img:
                    imge = Image.open(root + '/test/' + i +'/' + j).convert('L')  
                    imge = imge.resize((200, 91))
                    imge = np.array(imge)#（200,91）
                    self.label.append(index)
                    self.data.append(imge)
        self.label = torch.tensor(self.label, dtype=torch.int64)
        self.data = torch.tensor(self.data, dtype=torch.uint8)

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        img = Image.fromarray(img.numpy(), mode='L') 
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)



class FP(Dataset):

    def __init__(self, root,  train=True, transform=None,target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.label = []
        self.data = []

        if self.train:
            file_list_label = os.listdir(self.root + '/train')
            file_list_label.sort(key=lambda x: int(x[:]))  
            for index, i in enumerate(file_list_label):
                file_list_img = os.listdir( self.root + '/train/' + i)  
                for j in file_list_img:
                    imge = Image.open(root + '/train/' + i + '/' + j).convert('L')
                    imge = imge.resize((150, 150))
                    imge = np.array(imge)
                    self.label.append(index)
                    self.data.append(imge)
        else:
            file_list_label = os.listdir(self.root + '/test')
            file_list_label.sort(key=lambda x: int(x[:]))  
            for index, i in enumerate(file_list_label):
                file_list_img = os.listdir(self.root + '/test/' + i) 
                for j in file_list_img:
                    imge = Image.open(root + '/test/' + i +'/' + j).convert('L')  

                    imge = imge.resize((150, 150))
                    imge = np.array(imge)
                    self.label.append(index)
                    self.data.append(imge)
        self.label = torch.tensor(self.label, dtype=torch.int64)
        self.data = torch.tensor(self.data, dtype=torch.uint8)

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        img = Image.fromarray(img.numpy(), mode='L')  
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)