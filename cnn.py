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
            nn.Conv2d(1,16, kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (16, 400, 12)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,8, kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (8, 400, 12)
            # nn.Conv2d(16,16, kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (16, 400, 12)
            nn.ReLU(),
            nn.MaxPool2d((4,1)), # output shape: (16, 100, 12)

            # nn.BatchNorm2d(16),
            # nn.Conv2d(16,32, kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (32, 100, 12)
            nn.BatchNorm2d(8),
            nn.Conv2d(8,32, kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (32, 100, 12)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,32, kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (32, 100, 12)
            nn.ReLU(),
            nn.MaxPool2d((4,1)), # output shape: (32, 25, 12)

            nn.BatchNorm2d(32),
            nn.Conv2d(32,16 , kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (16, 25, 12)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,8 , kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (8, 25, 12)
            nn.ReLU(),
            nn.MaxPool2d((2,2)), # output shape: (8, 12, 6)

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

class CNN_Early(nn.Module):
    def __init__(self, number_gesture=49, class_rest=False, dropout=0.4):
        super(CNN_Early, self).__init__()
        output_class = number_gesture + int(class_rest==True)
        self.layers_early = nn.Sequential(
            nn.BatchNorm2d(1), # output shape: (1, 400, 12)
            nn.Conv2d(1,16, kernel_size=(3,3), stride=(1,1), padding='same', padding_mode='circular'), # output shape: (8, 400, 12)
            nn.ReLU(),
            nn.MaxPool2d((4,2)), # output shape: (64, 100, 6)
            
            nn.BatchNorm2d(16),
            nn.Conv2d(16,8, kernel_size=(3,3), stride=(1,1), padding='same', padding_mode='circular'), # output shape: (4, 100, 6)
            nn.ReLU(),
            nn.MaxPool2d((4,2)), # output shape: (8, 25, 3)

            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(600,128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_class)
        )

    def forward(self, x):
        num_batch, window_size, num_channel = x.shape
        x = x.view(num_batch,1,window_size, num_channel)
        # x = x.permute(0,2,1)
        x = self.layers_early(x)

        return x

class CNN_Early_Late(nn.Module):
    def __init__(self, number_gesture=49, class_rest=False, dropout=0.4, isEarlyExit=False):
        super(CNN_Early_Late, self).__init__()
        self.isEarlyExit = isEarlyExit
        output_class = number_gesture + int(class_rest==True)

        self.layers_common = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1,16, kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (16, 400, 12)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,8, kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (8, 400, 12)
            nn.ReLU(),
        )

        self.layers_early_feat_extractor = nn.Sequential(
            nn.MaxPool2d((4,2)), # output shape: (8, 100, 6)
            nn.BatchNorm2d(8),
            nn.Conv2d(8,4, kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (4, 100, 6)
            nn.ReLU(),
            nn.MaxPool2d((4,2)), # output shape: (4, 25, 3)

            nn.Flatten(),
            nn.BatchNorm1d(300),
            nn.Dropout(dropout),
        )
        self.layers_early2_classifier = nn.Sequential(
            nn.Linear(300,128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_class)
        )

        self.layers_late_feat_extractor = nn.Sequential(
            nn.MaxPool2d((4,1)), # output shape: (16, 100, 12)

            nn.BatchNorm2d(8),
            nn.Conv2d(8,32, kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (32, 100, 12)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,32, kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (32, 100, 12)
            nn.ReLU(),
            nn.MaxPool2d((4,1)), # output shape: (32, 25, 12)

            nn.BatchNorm2d(32),
            nn.Conv2d(32,16 , kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (16, 25, 12)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,8 , kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (8, 25, 12)
            nn.ReLU(),
            nn.MaxPool2d((2,2)), # output shape: (8, 12, 6)

            nn.Flatten(),
            nn.BatchNorm1d(576),
            nn.Dropout(dropout),
        )

        self.layers_late_feat_classifier = nn.Sequential(
            nn.Linear(576,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, output_class)
        )

    def forward(self, x):
        num_batch, window_size, num_channel = x.shape
        x = x.view(num_batch,1,window_size, num_channel)
        x = self.layers_common(x)
        if self.isEarlyExit:
            x_feat = self.layers_early_feat_extractor(x)
            y = self.layers_early_feat_classifier(x_feat)
        else:
            x_feat = self.layers_late_feat_extractor(x)
            y = self.layers_late_feat_classifier(x_feat)

        return y

# class CNN_Early_Late(nn.Module):
#     def __init__(self, number_gesture=49, class_rest=False, dropout=0.4, isEarlyExit=False):
#         super(CNN_Early_Late, self).__init__()
#         self.isEarlyExit = isEarlyExit
#         output_class = number_gesture + int(class_rest==True)
#         self.layers_common = nn.Sequential(
#             nn.BatchNorm2d(1), # output shape: (1, 400, 12)
#             nn.Conv2d(1,16, kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (16, 400, 12)
#             nn.ReLU(),
#             nn.MaxPool2d((4,2)), # output shape: (64, 100, 6)
            
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16,8, kernel_size=(3,3), stride=(1,1), padding='same'), # output shape: (8, 100, 6)
#             nn.ReLU(),
#         )

#         self.layers_early1 = nn.Sequential(
#             nn.MaxPool2d((4,2)), # output shape: (8, 25, 3)
#             nn.Flatten(),
#             nn.Dropout(dropout),
#         )
#         self.layers_early2 = nn.Sequential(
            
#             nn.Linear(600,128),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(128, output_class)
#         )

#         self.layers_late = nn.Sequential(
#             nn.BatchNorm2d(8),
#             nn.Conv2d(8,32, kernel_size=(3,3), stride=(1,1), padding='same', padding_mode='circular'), # output shape: (32, 100, 6)
#             nn.ReLU(),
#             nn.MaxPool2d((2,2)), # output shape: (32, 50, 3)

#             nn.BatchNorm2d(32),
#             nn.Conv2d(32,32, kernel_size=(3,3), stride=(1,1), padding='same', padding_mode='circular'), # output shape: (32, 50, 3)
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32,8, kernel_size=(3,3), stride=(1,1), padding='same', padding_mode='circular'), # output shape: (32, 50, 3)
#             nn.ReLU(),

#             nn.MaxPool2d((2,1)), # output shape: (8, 25, 3)
#             nn.Flatten(),
#             nn.Dropout(dropout),
#             nn.Linear(600,128),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(128, output_class)
#         )

#     def forward(self, x):
#         num_batch, window_size, num_channel = x.shape
#         x = x.view(num_batch,1,window_size, num_channel)
#         x = self.layers_common(x)
#         if self.isEarlyExit:
#             x_feat = self.layers_early1(x)
#             x = self.layers_early2(x_feat)
#         else:
#             x = self.layers_late(x)

#         return x