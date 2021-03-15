# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 21:34:51 2021

@author: Hostl
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from sklearn import svm
from sklearn.metrics import accuracy_score
from multiprocessing import Process, freeze_support
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import time


if __name__ == '__main__':
    start_time = time.time()
    freeze_support()