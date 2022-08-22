# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:30:52 2022

@author: Anish Hilary
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cnn_data

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Custom_data (Dataset):
    def __init__(self, data, label, transform):
        super(Dataset,self).__init__()
        self.data = data
        self.label = torch.LongTensor(label)
        self.transform = transform
        
    def __len__(self):
        len_data = len(self.data)
        # self.len_label = len(self.label)
        return len_data
        
    def __getitem__(self, index):
        x = self.data[index]
        x = self.transform(x)
        y = self.label[index]
        
        return x,y
    

    
    
transform, data, label = cnn_data.all_data()
train_set = Custom_data(data, label, transform)


def train():
    return train_set
    



# imshow(torchvision.utils.make_grid(imag))

