# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:21:08 2022

@author: Anish Hilary
"""

import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt




img =  os.listdir(os.path.join(os.getcwd(),"Images"))

cnn_im = []
cnn_lab = []



def svm_iou(gnd_tru,bound_box):
    a1,b1,a2,b2 = gnd_tru
    a3,b3,a4,b4 = bound_box       
#intersection
    y_b = min(b2,b4)         
    y_t = max(b1,b3)
    x_l = max(a1,a3)
    x_r = min(a2,a4)  
# check for intersection
    if x_r<x_l or y_t>y_b:
        return 0
    
    else:
# Union and intersection area
        area1 = (a2-a1)*(b2-b1)          
        area2 = (a4-a3)*(b4-b3)
        inter_area = (x_r - x_l)*(y_b - y_t)
        union_area = area1+area2-inter_area
#IOU   
        iou_area = (inter_area)/float(union_area)
        assert iou_area>=0.0
        assert iou_area<=1.0
        return iou_area
    
    


for i,pic in enumerate(img):
    try:
        if pic.startswith ("airplane_003"):
            gnd_tru = []
        
# collecting ground truth
            cv_img = cv2.imread(os.path.join(os.getcwd(),"Images",pic))
            df = pd.read_csv(os.path.join(os.getcwd(), "Airplanes_Annotations",pic.replace(".jpg",".csv")))
            for j in range(len(df)):
                x1 = int(df.iloc[j,0].split(" ")[0])
                y1 = int(df.iloc[j,0].split(" ")[1])
                x2 = int(df.iloc[j,0].split(" ")[2])
                y2 = int(df.iloc[j,0].split(" ")[3])
                cv2.rectangle(cv_img,(x1,y1),(x2,y2),(255,0,0),1)
                gnd_tru.append([x1,y1,x2,y2])
                timage = cv_img[y1:y2,x1:x2]
                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
            
# selective search                                                                                
            cv2.setUseOptimized(True)
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss.setBaseImage(cv_img)
            ss.switchToSelectiveSearchFast()
            rects = ss.process()
            img_cop = cv_img.copy()
            
          
            falsecounter = 0
            counter = 0
            flag = 0
            for gt in gnd_tru:
                for e,result in enumerate(rects):
                    if e < 2000 and flag == 0:
                            x,y,w,h = result
                            bb = [x,y,x+w,y+h]
                            iou_area = svm_iou(gt,bb)
                            if (counter < 16 and iou_area > 0.7):
                                    timage = img_cop[y:y+h,x:x+w]
                                    resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                    cnn_im.append(resized)
                                    cnn_lab.append(1)
                                    counter += 1  
                            if (falsecounter < 27 and iou_area < 0.3):
                                    fimage = img_cop[y:y+h,x:x+w]
                                    resized = cv2.resize(fimage, (224,224), interpolation = cv2.INTER_AREA)
                                    cnn_im.append(resized)
                                    cnn_lab.append(0)
                                    falsecounter += 1
                            if (counter == 15 and falsecounter == 16):
                                flag = 1

            print(f'image: {pic}')
    except Exception as e:
        print(e)
        print("error in "+ pic)
        continue



cnn_image = np.array(cnn_im)
cnn_label = np.array(cnn_lab)
print("all data collected")
from torchvision import transforms

def all_data():
    
    print("started data collection")

    img_transform = {"train" : transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        ])}
    
    trans = img_transform["train"]
    
    return trans,cnn_image,cnn_label

c=0
for i in cnn_image:
    plt.figure(c)
    plt.imshow(i)
    c+=1
    
print(f'the cnn label {cnn_label}')