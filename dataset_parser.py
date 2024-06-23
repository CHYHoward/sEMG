import numpy as np 
import matplotlib.pyplot as plt

import argparse
import multiprocessing as mp
import os
import scipy.signal as signal

# Torch
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader

# User-defined
from mat2np_segment_all_subject import *
from dsp_preprocess import *
from feature_extractor import *

def dataset_filter_normalize_segementation(fileName, fs=2000, window_size = 400, window_step=200, num_channel=12, type_filter="BPF_20_200", type_norm="mvc"):
    gesture_label  = np.array(int(fileName.split('/')[-1].split('.')[0].split('_')[1]))
    emg_sample = np.load(fileName)

    # Filtering and Normalization
    emg_sample_filter = filter(emg_sample,type_filter=type_filter, fs=fs, order=1)    # whole_window_size * num_channel (e.g., LPF_200_, HPF_20_, ...)
    # emg_sample_filter_norm = normalization(emg_sample_filter, type_norm="standardization")
    emg_sample_filter_norm = normalization(emg_sample_filter, type_norm=type_norm)

    # Sliding window segmentation
    num_window = (np.floor((emg_sample_filter_norm.shape[0]-window_size)/window_step) + 1).astype(int)
    # emg_sample_filter_norm_seg_batch = None
    # gesture_label_batch = None
    emg_sample_filter_norm_seg_batch = np.zeros((num_window, window_size, num_channel))
    gesture_label_batch = np.zeros((num_window, 1))

    for i in range(num_window):
        emg_sample_filter_norm_seg = emg_sample_filter_norm[i*window_step:i*window_step+window_size]
        # emg_sample_filter_norm_seg_batch = handle_concatenation(emg_sample_filter_norm_seg_batch, emg_sample_filter_norm_seg.reshape(1,window_size,-1), axis=0)
        # gesture_label_batch = handle_concatenation(gesture_label_batch, gesture_label.reshape(1,-1), axis=0)
        
        num_sample = emg_sample_filter_norm_seg.shape[0]
        emg_sample_filter_norm_seg_batch[i] = emg_sample_filter_norm_seg
        gesture_label_batch[i] = gesture_label

    return emg_sample_filter_norm_seg_batch, gesture_label_batch  #shape: (num_window, window_size, num_channel); #shape: (num_window,1)

def dataset_filter_normalize_segementation_all_subject_exercise(dataset_type="train", subject_list = [i+1 for i in range(40)], \
                                                                exercise_list = [1,2,3], fs=2000, window_size = 400, window_step=200, num_channel=12, class_rest=False, \
                                                                type_filter = "none", type_norm = "mvc"):
    if dataset_type == "test":
        trial_list = [2,5]
    else: 
        trial_list = [1,3,4,6]
    
    # Preallocate memory for speed up
    num_sample_max = int(1e6)
    emg_sample_dataset = np.zeros((num_sample_max, window_size, num_channel))
    gesture_label_dataset = np.zeros((num_sample_max))
    cnt = 0
    # print('', end='', flush=True)
    for i, idx_subject in enumerate(subject_list):
        # start = time.time()
        end_tpye = '\n' if i==len(subject_list)-1 else '\r'
        # end_tpye = '\n'
        print("Loading %2d-th subject for %s dataset ..." %(idx_subject, dataset_type), end=end_tpye, flush=True)
        for idx_exercise in exercise_list:
            for idx_quarter, idx_trial in enumerate(trial_list):
                PATH_seg_np = "Dataset/DB2/DB2_np/DB2_s_{}/exercise_{}/trial_{}/".format(idx_subject,idx_exercise,idx_trial)
                fileNames = [PATH_seg_np+i for i in os.listdir(PATH_seg_np)]
                for fileName in fileNames:
                    
                    [emg_sample_filter_norm_seg_batch, gesture_label_batch] = dataset_filter_normalize_segementation(\
                        fileName, fs=fs, window_size=window_size, window_step=window_step, num_channel=num_channel,type_filter = type_filter, type_norm = type_norm)
                    
                    num_window = gesture_label_batch.shape[0]
                    num_channel = emg_sample_filter_norm_seg_batch.shape[2]

                    if dataset_type != "test":
                        num_window_valid = np.floor(num_window/4).astype(int)   # one certain quarter for validation
                        idx_all = range(num_window)
                        idx_valid = range(idx_quarter*num_window_valid,(idx_quarter+1)*num_window_valid)
                        idx_train = [i for i in idx_all if i not in idx_valid]
                        if dataset_type == "train":
                            idx_datatype = idx_train
                        elif dataset_type == "valid":
                            idx_datatype = idx_valid

                        emg_sample_filter_norm_seg_batch = emg_sample_filter_norm_seg_batch[idx_datatype]
                        gesture_label_batch = gesture_label_batch[idx_datatype]

                    num_sample = emg_sample_filter_norm_seg_batch.shape[0]
                    emg_sample_dataset[cnt:cnt+num_sample] = emg_sample_filter_norm_seg_batch

                    if exercise_list == [2]:
                        gesture_label_batch = gesture_label_batch-17
                    elif exercise_list == [3]:
                        gesture_label_batch = gesture_label_batch -17 -23
                    gesture_label_dataset[cnt:cnt+num_sample] = np.squeeze(gesture_label_batch-1+int(class_rest)) # Counting the label from 0 if "rest" class is not considered

                    cnt += num_sample   # counter to indicate the size of current dataset

        # end = time.time()
        # print("Elasped time: ", end - start)
    
    emg_sample_dataset = emg_sample_dataset[0:cnt]
    gesture_label_dataset = gesture_label_dataset[0:cnt]
    
    return emg_sample_dataset, gesture_label_dataset

class DB2_Dataset(Dataset):
    def __init__(self, dataset_type="train", subject_list = [i+1 for i in range(40)], exercise_list = [1,2,3],fs=2000, \
                window_size = 400, window_step=200, num_channel=12, feat_extract=False, class_rest=False, type_filter = "none", type_norm = "mvc",\
                load_dataset=True, save_dataset=False):
        
        if load_dataset:
            print(f"Loading saved {dataset_type} dataset in ./Dataset/DB2/")
            emg_sample_dataset = np.load(f"Dataset/DB2/emg_sample_dataset_{dataset_type}.npy")
            gesture_label_dataset = np.load(f"Dataset/DB2/gesture_label_dataset_{dataset_type}.npy")
        else:
            emg_sample_dataset, gesture_label_dataset = dataset_filter_normalize_segementation_all_subject_exercise(\
                dataset_type=dataset_type,subject_list=subject_list, exercise_list=exercise_list, fs=fs, window_size=window_size, window_step=window_step, \
                num_channel=num_channel,class_rest=class_rest, type_filter = type_filter, type_norm = type_norm)
            if save_dataset:
                np.save(f"Dataset/DB2/emg_sample_dataset_{dataset_type}",emg_sample_dataset)
                np.save(f"Dataset/DB2/gesture_label_dataset_{dataset_type}",gesture_label_dataset)

        self.emg_sample_dataset = emg_sample_dataset
        self.gesture_label_dataset = gesture_label_dataset

        if feat_extract:
            self.emg_sample_dataset = feat_Hudgins_FS(self.emg_sample_dataset).reshape(self.emg_sample_dataset.shape[0],-1) # shape: (num_batch, num_feature * num_channel)
        # else: self.emg_sample_dataset shape: (num_batch, window_size, num_channel)

    def __len__(self):
        return self.gesture_label_dataset.shape[0]
    
    def __getitem__(self, idx):
        return (torch.tensor(self.emg_sample_dataset[idx], dtype=torch.float), torch.tensor( self.gesture_label_dataset[idx], dtype=torch.long))

def train_test_split_DataLoader(batch_size = 256, subject_list = [i+1 for i in range(40)], exercise_list=[1,2,3], fs=2000, \
                                window_size = 400, window_step=200, num_channel=12, feat_extract = False, class_rest = False, \
                                type_filter = "none", type_norm = "none", load_dataset = False, save_dataset = False):
    print("\n"+"-"*70)
    # Load the DB2 datase 
    dataset_train = DB2_Dataset(dataset_type="train", subject_list=subject_list, feat_extract=feat_extract, exercise_list=exercise_list, fs=fs, \
                                window_size=window_size, window_step=window_step, num_channel=num_channel, class_rest = class_rest, \
                                type_filter = type_filter, type_norm = type_norm, load_dataset=load_dataset, save_dataset=save_dataset)
    dataset_valid = DB2_Dataset(dataset_type="valid", subject_list=subject_list, feat_extract=feat_extract, exercise_list=exercise_list, fs=fs, \
                                window_size=window_size, window_step=window_step, num_channel=num_channel, class_rest = class_rest, \
                                type_filter = type_filter, type_norm = type_norm, load_dataset=load_dataset, save_dataset=save_dataset)
    dataset_test  = DB2_Dataset(dataset_type="test",  subject_list=subject_list, feat_extract=feat_extract, exercise_list=exercise_list, fs=fs, \
                                window_size=window_size, window_step=window_step, num_channel=num_channel, class_rest = class_rest, \
                                type_filter = type_filter, type_norm = type_norm, load_dataset=load_dataset, save_dataset=save_dataset)
    print("-"*70)
    print("Number of train data: %5d" %(len(dataset_train)))
    print("Number of valid data: %5d" %(len(dataset_valid)))
    print("Number of test  data: %5d" %(len(dataset_test)))
    print("-"*70+"\n")

    # DataLoader
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    emg_sample_dataset, gesture_label_dataset = dataset_filter_normalize_segementation_all_subject_exercise()
    print(emg_sample_dataset.shape)
    print("MVC value: ", emg_sample_dataset.max())