# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:09:24 2022

@author: Anish Hilary
"""


import torch
import torch.nn as nn
import torchvision.models as models
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms



mod = models.vgg16()


num_in = mod.classifier[-1].in_features                                                        # getting the input of last fc layer
new_classifier = list(mod.classifier.children())[:-1]                                          # slicing all the fc layer except the last
new_layer = [nn.Linear(num_in, 256),nn.ReLU(inplace=True),nn.Dropout(0.4),nn.Linear(256, 1), nn.Sigmoid()]   # creating new layer
                 
new_classifier.extend(new_layer)                                                               # joining the new layer as last layer
mod.classifier = nn.Sequential(*new_classifier)                                                # converting list to nn.Seq



mod.load_state_dict(torch.load('new_wgts.pth'))

mod.eval()


for param in mod.parameters():
    param.requires_grad = False


img = os.path.join(os.getcwd(),"Images","airplane_001.jpg")


# for img in img_list:
#     if not img.startswith("airplane"):
#         print(img)

cv_img = cv2.imread(img)

# selective search

cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(cv_img)
ss.switchToSelectiveSearchFast()
rects = ss.process()
img_cop = cv_img.copy()

box = []
c = 0
maxi = 0
for e,rec in enumerate(rects):
    if c<2000:
        c+=1
        x,y,w,h = rec
        print(x,y,w,h)
        timg = img_cop[x:x+w,y:y+h]
        resized = cv2.resize(timg,(224,224),interpolation = cv2.INTER_AREA)
        transform = transforms.Compose([
transforms.ToTensor()
])
        resized = transform(resized)
        resized = torch.unsqueeze(resized,0)
        output = mod(resized)
        out = output.detach().squeeze(-1)
        print(f'Squeezed shape is {out.shape} with out value {out}')       
        if out > 0.4:
            maxi = max(maxi,out)
            print(f'More than 0.5 value is {out}')
            box.extend([rec])

print(box)
print(f'maximum is {maxi}')
for e,i in enumerate(box):
    x,y,w,h = i
    cv2.rectangle(img_cop,(x,y),(x+w,y+h),(0,255,0),1,cv2.LINE_AA)
    plt.figure(e)
    plt.imshow(img_cop)    
    