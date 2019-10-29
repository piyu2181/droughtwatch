#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:43:26 2019

@author: debjani
"""

'''
This is the main code to train

'''
import glob
import torch
import torch.nn as nn
import numpy as np
import argparse
import torch.backends.cudnn
from torch.optim import Adam
from dataset import Droughtwatch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models

import torchvision 
from transformsdata import (DualCompose,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        Rotate,
                        CenterCrop,
                        VerticalFlip)

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--batch-size', type=int, default=1)
    arg('--limit', type=int, default=100, help='number of images in epoch')
    arg('--num_epochs1', type=int, default=20)
    arg('--num_epochs2', type=int, default=1)
    arg('--model', type=str, default='Resnet50')

    args = parser.parse_args()

    num_classes = 4
    
    #############################################################################################################
    if args.model == 'Resnet50':
        model = torchvision.models.resnet50(pretrained=False)
    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None

    PATH = "/home/bhowmicd/codes/droughtwatch/src/resnet50-19c8e357.pth"
    num_classes = 4
    cudnn.benchmark = True
    model = torchvision.models.resnet50(pretrained=False)
    model.load_state_dict(torch.load(PATH))
    #model = nn.parallel.DataParallel(model, device_ids=range(1))
    device = torch.device("cuda:0")
    
    model.to(device)
    #############################################################################################################
    train_image_path= '/home/bhowmicd/projects/rpp-bengioy/drougtwatch/data/train/images/'
    val_image_path = '/home/bhowmicd/projects/rpp-bengioy/drougtwatch/data/val/images/'
    get_train_path = train_image_path + "/*.npy"
    train_image_file = np.array(sorted(glob.glob(get_train_path)))
    print("train_file: ", len(train_image_file))
    get_val_path = val_image_path + "/*.npy"
    val_image_file = np.array(sorted(glob.glob(get_val_path)))
    print("val_file: ", len(val_image_file))
    #################################################################################################################
    def make_loader(file_names, shuffle=False, transform=None, limit=None, batch_size = 4, mode = "train") :
             return DataLoader(
                dataset= Droughtwatch(file_names, transform=transform, limit=limit, mode = mode),
                shuffle=shuffle,
                batch_size= batch_size)
    #################################################################################################################
    train_transform = DualCompose([
            CenterCrop(64),
            HorizontalFlip(),
            VerticalFlip(),
            Rotate(),
            ImageOnly(Normalize())
        ])
    
    val_transform = DualCompose([
            CenterCrop(64),
            ImageOnly(Normalize())
        ])
    train_loader = make_loader(train_image_file, shuffle=True, transform=train_transform,  batch_size = 128, mode = "train")
    val_loader = make_loader(val_image_file, transform= val_transform,  batch_size = 32 , mode = "val")
    #################################################################################################################
    def run_epoch(model, loss_fn, loader, optimizer, dtype):
        """
        Train the model for one epoch.
        """
        # Set the model to training mode
        model.train()
        counter = 0
        device = torch.device("cuda:0")
        model.to(device)
        for x_var, y_var in loader:
            counter +=1
            x_var = Variable(x_var.type(dtype))
            y_var = Variable(y_var.type(dtype).long())
        
            
            x_var = x_var.to(device)
            y_var = y_var.to(device)
            # Run the model forward to compute scores and loss.
            scores = model(x_var)
            #print("scores:", scores)
            loss = loss_fn(scores, y_var)
            print(str(counter)+'/'+str(len(loader)) + ';    Loss : ' + str(loss.item()))
            # Run the model backward and take a step using the optimizer.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    
    
    ############################################################################################################
    def check_accuracy(model, loader, dtype):
        """
        Check the accuracy of the model.
        """
        # Set the model to eval mode
        model.eval()
        num_correct, num_samples = 0, 0
        device = torch.device("cuda:0")
        with torch.no_grad():
            for x_var, y_var in loader:
                
                x_var = Variable(x_var.type(dtype))
                x_var = x_var.to(device)
                y_var = y_var.to(device)
                scores = model(x_var)
                _, preds = scores.data.cpu().max(1)
                #print("preds:", preds)
                preds = preds.to(device)
                num_correct += (preds == y_var).sum()
                num_samples += x_var.shape[0]

            acc = float(num_correct) / num_samples
            return acc
    ##################################################################################################################    
    
    
    

    file1 = open("/home/bhowmicd/codes/droughtwatch/src/models/train_losses.txt", "w+")
    #file2 = open("/home/bhowmicd/codes/droughtwatch/src/models/train_losses.txt", "w+")
    dtype = torch.FloatTensor

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    model.type(dtype)
    weights = torch.FloatTensor([1., 4., 4., 6.])
    weights = weights.to(device)
    loss_fn = nn.CrossEntropyLoss(weight = weights)
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Construct an Optimizer object for updating the last layer only.
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-2)
    for epoch in range(args.num_epochs1):
    # Run an epoch over the training data.
        print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
        run_epoch(model, loss_fn, train_loader, optimizer, dtype)
    
        print('Calculating accuracies')
        train_acc = 0
        val_acc = check_accuracy(model, val_loader, dtype)
        print('Train accuracy: ', train_acc, ';    Val accuracy: ', val_acc)
        file1.write("{}:{}".format(train_acc, ", ", val_acc))
    
    torch.save(model.module.state_dict(), "home/bhowmicd/codes/droughtwatch/src/models/model_10.pth")
if __name__ == '__main__':
    main()

