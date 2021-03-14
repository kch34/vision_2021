# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:57:18 2021

@author: Austin Tercha
"""

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


FISH_DIR = "fish_up_close"

def data_normalization():
    # Data normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(FISH_DIR, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device, class_names, dataloaders, dataset_sizes


def train_model(model, num_epochs, dataloaders, device, optimizer,
                criterion, scheduler, dataset_sizes):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
      for phase in ['train', 'val']:
        if phase == 'train':
          model.train()
        else:
          model.eval()
    
        for inputs, labels in dataloaders[phase]:
          inputs = inputs.to(device)
          labels = labels.to(device)
    
        optimizer.zero_grad()
    
        all_batchs_loss = 0
        all_batchs_corrects = 0
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)
          if phase == 'train':
            loss.backward()
            optimizer.step()
          all_batchs_loss += loss.item() * inputs.size(0)
          all_batchs_corrects += torch.sum(preds == labels.data)
        if phase == 'train':
          scheduler.step()
        epoch_loss = all_batchs_loss / dataset_sizes[phase]
        epoch_acc = all_batchs_corrects.double() / dataset_sizes[phase]
        if phase == 'val' and epoch_acc > best_acc:
          best_acc = epoch_acc
          best_model_wts = copy.deepcopy(model.state_dict())
          torch.save(best_model_wts , 'best_model_weight.pth')


if __name__ == "__main__":
    # Data normalization
    device, class_names, dataloaders, dataset_sizes = data_normalization()

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, 2)
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Model training
    train_model(model, 25, dataloaders, device, 
                optimizer, criterion, scheduler, dataset_sizes)
    
