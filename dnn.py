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