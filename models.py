import numpy as np 
import matplotlib.pyplot as plt

# Torch
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Other
import argparse
import multiprocessing as mp
import os
import scipy.signal as signal

class DNN_feature(nn.Module):
    def __init__(self, number_gesture=49, class_rest=False, dropout=0.5):
        super(DNN_feature, self).__init__()
        output_class = number_gesture + int(class_rest==True)
        self.layers = nn.Sequential(
            nn.BatchNorm1d(59),
            nn.Linear(59, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_class)
        )

    def forward(self, x):
        return self.layers(x)

class DNN1_feature(nn.Module):
    def __init__(self, number_gesture=49, class_rest=False, dropout=0.5):
        super(DNN1_feature, self).__init__()
        output_class = number_gesture + int(class_rest==True)
        self.layers = nn.Sequential(
            nn.BatchNorm1d(59),
            nn.Linear(59, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_class)
        )

    def forward(self, x):
        return self.layers(x)

class DNN2_feature(nn.Module):
    def __init__(self, number_gesture=49, class_rest=False, dropout=0.5):
        super(DNN2_feature, self).__init__()
        output_class = number_gesture + int(class_rest==True)
        self.layers = nn.Sequential(
            nn.BatchNorm1d(59),
            nn.Linear(59,128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
                        
            nn.Linear(128, output_class)
        )

    def forward(self, x):
        return self.layers(x)

class DNN3_feature(nn.Module):
    def __init__(self, number_gesture=49, class_rest=False, dropout=0.5):
        super(DNN3_feature, self).__init__()
        output_class = number_gesture + int(class_rest==True)
        self.layers = nn.Sequential(
            nn.BatchNorm1d(59),
            nn.Linear(59,64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
                        
            nn.Linear(64, output_class)
        )

    def forward(self, x):
        return self.layers(x)
    
class CNN(nn.Module):
    def __init__(self, number_gesture=49, class_rest=False, dropout=0.4):
        super(CNN, self).__init__()
        output_class = number_gesture + int(class_rest==True)
        self.layers = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1,64, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((4,1)),
            # nn.BatchNorm2d(32),
            nn.Conv2d(64,64, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((4,1)),
            # nn.BatchNorm2d(16),
            nn.Conv2d(64,8 , kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(576,128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_class)
            # nn.Linear(64, output_class)
        )
        self.conv = nn.Conv2d(1, 16, kernel_size=(3,3), stride=(1,1), padding='same')
        self.maxpool = nn.MaxPool2d((4,1))

    def forward(self, x):
        num_batch, window_size, num_channel = x.shape
        x = x.view(num_batch,1,window_size, num_channel)
        return self.layers(x)

class CNN1(nn.Module):
    def __init__(self, number_gesture=49, class_rest=False, dropout=0.4):
        super(CNN1, self).__init__()
        output_class = number_gesture + int(class_rest==True)
        self.layers = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1,64, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((4,1)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((4,1)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,8 , kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Flatten(),
            nn.BatchNorm1d(576),
            nn.Dropout(dropout),
            nn.Linear(576,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, output_class)
            # nn.Linear(64, output_class)
        )
        self.conv = nn.Conv2d(1, 16, kernel_size=(3,3), stride=(1,1), padding='same')
        self.maxpool = nn.MaxPool2d((4,1))

    def forward(self, x):
        num_batch, window_size, num_channel = x.shape
        x = x.view(num_batch,1,window_size, num_channel)
        return self.layers(x)

class CNN2(nn.Module):
    def __init__(self, number_gesture=49, class_rest=False, dropout=0.4):
        super(CNN2, self).__init__()
        output_class = number_gesture + int(class_rest==True)
        self.layers = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1,16, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,16, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((4,1)),

            nn.BatchNorm2d(16),
            nn.Conv2d(16,32, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,32, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((4,1)),

            nn.BatchNorm2d(32),
            nn.Conv2d(32,16 , kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,8 , kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Flatten(),
            nn.BatchNorm1d(576),
            nn.Dropout(dropout),
            nn.Linear(576,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, output_class)
            # nn.Linear(64, output_class)
        )
        self.conv = nn.Conv2d(1, 16, kernel_size=(3,3), stride=(1,1), padding='same')
        self.maxpool = nn.MaxPool2d((4,1))

    def forward(self, x):
        num_batch, window_size, num_channel = x.shape
        x = x.view(num_batch,1,window_size, num_channel)
        return self.layers(x)

class CNN2_Early(nn.Module):
    def __init__(self, number_gesture=49, class_rest=False, dropout=0.4):
        super(CNN2, self).__init__()
        output_class = number_gesture + int(class_rest==True)
        self.layers = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1,16, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((8,1)),

            nn.Conv2d(16,8, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Flatten(),
            nn.BatchNorm1d(576),
            nn.Dropout(dropout),
            nn.Linear(576,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, output_class)
            # nn.Linear(64, output_class)
        )
        self.conv = nn.Conv2d(1, 16, kernel_size=(3,3), stride=(1,1), padding='same')
        self.maxpool = nn.MaxPool2d((4,1))

    def forward(self, x):
        num_batch, window_size, num_channel = x.shape
        x = x.view(num_batch,1,window_size, num_channel)
        return self.layers(x)