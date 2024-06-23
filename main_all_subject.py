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
from torchsummary import summary

# Save/Load as mat.
from scipy.io import savemat, loadmat
import h5py
import mat73 # https://github.com/skjerns/mat7.3

# Progress Bar
from rich.progress import track
from rich.progress import Progress

# Other
import argparse
import multiprocessing as mp
import os
import scipy.signal as signal
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
from plot import *

# User-defined
from mat2np_segment_all_subject import *
from dsp_preprocess import *
from dataset_parser import *
from models import *
from feature_extractor import *
from set_args import *
from train_test_process import *
from vit import *

set_seed(87)

# Parameter setup
args = get_args()
print_args(args)
window_size, window_step, number_gesture, model_PATH, device = get_args_info(args)

# Dataset setup
train_loader, valid_loader, test_loader = train_test_split_DataLoader(\
                                        batch_size=args.batch_size, subject_list=args.subject_list, exercise_list= args.exercise_list, \
                                        fs=args.fs, window_size=window_size, window_step=window_step, num_channel=args.num_channel, \
                                        feat_extract=args.feat_extract, class_rest=args.class_rest, type_filter = args.type_filter, type_norm = args.type_norm, \
                                        load_dataset = args.load_dataset, save_dataset = args.save_dataset)

if "ViT" in args.model_type:
    # model = ViT_TNet(window_size, args.num_channel).to(device) # TNet
    model = eval(f"{args.model_type}(window_size, args.num_channel,number_gesture=number_gesture, class_rest=args.class_rest)").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=1e-5, verbose=False)
    criterion = nn.CrossEntropyLoss()
else:
    model = eval(f"{args.model_type}(number_gesture=number_gesture, class_rest=args.class_rest)").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = None
    criterion = nn.CrossEntropyLoss()  

# Model Training and Validation
if args.en_train:
    train_process(args.num_epoch,model,model_PATH,train_loader,valid_loader,device,optimizer,criterion, scheduler=scheduler)

# Model Testing
y_pred, y_gold, acc_test = test_process(model,model_PATH,test_loader,device,criterion,args.model_type, args.load_model)

# Plot confusion matrix
plot_cm(y_gold, y_pred, args.log_name)

