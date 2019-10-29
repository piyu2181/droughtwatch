#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 20:31:29 2019

@author: debjani
"""

import torch
import numpy as np
from torch.utils.data import Dataset



class Droughtwatch(Dataset):
    def __init__(self, img_paths: list, transform=None, mode='train', limit=None):
        self.img_paths = img_paths
        self.transform = transform
        self.mode = mode
        #print(self.mode)
        self.limit = limit
        
    def __len__(self):
        if self.limit is None:
            return len(self.img_paths)
        else:
            return self.limit 
    
    def __getitem__(self, idx): 
        if self.limit is None:
            img_file_name = self.img_paths[idx]

        else:
            img_file_name = np.random.choice(self.img_paths)
            #print(img_file_name)
        
        img = load_image(img_file_name)
        
        idx = img_file_name[-9:-4]
        if self.mode == "train":
            strg = "train"
        else:
            strg = "val"
        file = img_file_name[:-23] + "/label/" + strg + "_labels.txt"
        #print(strg)
        #print(file)
        file = open(file)
        for line in file:
            index = str(line[0:5])
            if index == str(idx):
                label = int(line[6])
    
        
        if self.mode == 'train':
               
            #label = load_label(self,img_file_name)  
            #label = label
            #print("train_label:",label)
            img = self.transform(img)    #cargo la img y transformo   
            return to_float_tensor(img),  label
            
        else:
            label_batch = label
            #print("val_label:", label_batch)
            img = self.transform(img)
            return to_float_tensor(img), label_batch
            


def load_image(path):
    img = np.load(str(path))
    #img=img.transpose((1, 2, 0)) 
    return  img 


def to_float_tensor(img):#(512, 512, 3)
    #print('img_tensor',img.shape)
    #print('img_tensor-to tensor bedore',img.shape)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img=torch.tensor(np.moveaxis(img, -1, 0), device=device).float()  #2,0,1 torch.Size([3, 512, 512]) convert to tensory change channels
    #print('img_tensor',img.shape) 
    return img 