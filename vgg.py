# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 11:54:17 2022

@author: Anish Hilary
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import time
import numpy
import model_data
import numpy as np
from sklearn.metrics import accuracy_score


mod = models.vgg16(pretrained=True)
mod.load_state_dict(torch.load('vgg16-397923af.pth'))


c=0
for param in mod.parameters():
    if c==24 or c==25:
        param.requires_grad = True
    else:
        param.requires_grad = False
    c+=1

      
num_in = mod.classifier[-1].in_features                                                        # getting the input of last fc layer
new_classifier = list(mod.classifier.children())[:-1]                                          # slicing all the fc layer except the last
new_layer = [nn.Linear(num_in, 256), nn.ReLU(inplace = True), nn.Dropout(p=0.4) ,nn.Linear(256,1), nn.Sigmoid()]   # creating new layer
                 
new_classifier.extend(new_layer)                                                               # joining the new layer as last layer
mod.classifier = nn.Sequential(*new_classifier)                                                # converting list to nn.Seq

for param in mod.parameters():
    if param.requires_grad == True:
        print(param.shape)

              
def my_mod():
    return mod

print(mod)

# getting training data from model_data module
train_set = model_data.train()
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 30, shuffle= False)

print(f'lenght of trainset is {len(train_set)}')
#SVM loss
class SVM_Loss(nn.modules.Module):    
    def __init__(self):
        super(SVM_Loss,self).__init__()
    def forward(self, outputs, labels):
          return torch.sum(torch.clamp(1 - outputs.t()*labels, min=0))


# Loss function and Optimizers 
loss_fn = nn.BCELoss()
optimizer = optim.Adam(mod.parameters(), lr=0.001, weight_decay=0.001)


# TRAIN

elaps = []
loss_per_epo = []
accu_per_epo = []
mod.train()
for epoch in range(10):
    loss_epo = []
    pred_epo = []
    targ_epo = []
    start = time.time()
    for e, (data,target) in enumerate(train_loader):
        
        # data,target = data.to(device),target.to(device)
        pred = mod(data)
        target = target.float()
        # print(pred.dtype, target.dtype)
        # print(f'The prediction is {pred} with shape {pred.shape} but target is {target} with shape {target.shape}')
        loss = loss_fn (pred,target.unsqueeze(1))
        # print(f'BUT THE AFTER SHAPES ARE PRED: {pred.shape} and TARGET: {target.shape}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predict = np.round(pred.detach())
        tar = np.round(target.detach())
        pred_epo.extend(predict.reshape(-1).tolist())
        targ_epo.extend(tar.tolist())
        loss_epo.append(loss.detach().numpy())
        print(f'step{e}, current epoch {epoch}')

    accuracy = accuracy_score(targ_epo,pred_epo)
    print(f'the accuracy is {accuracy}')
    stop = time.time()
    elap = stop - start
    elaps.append(elap)
    lo_epo = numpy.mean(loss_epo)
    print(f'loss in {epoch} is {lo_epo}')
    loss_per_epo.append(lo_epo)
    accu_per_epo.append(accuracy)

# PLOTS 

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(loss_per_epo)  
plt.xlabel("no. of epochs")
plt.ylabel("loss")
plt.title("loss vs epoch")

print(f'The accuracy at each epoch is : {accu_per_epo}')

plt.figure(2)
plt.plot(accu_per_epo)
plt.xlabel("no. of epochs")
plt.ylabel("accuracy")
plt.title("accuracy vs epoch")

print(f'The loss at each epoch is " {loss_per_epo}')

plt.figure(3)
plt.plot(elaps)
plt.xlabel("no. of epochs")
plt.ylabel("time taken")
plt.title("time vs epoch")

torch.save(mod.state_dict(), 'new_wgts.pth')

