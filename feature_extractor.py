import numpy as np 
import matplotlib.pyplot as plt

# Other
import argparse
import multiprocessing as mp
import os
import scipy.signal as signal

# Some feature extraction

# input x : numpy, shape: (num_batch, window_size, num_channel)
# output y: numpy, shape: (num_batch, num_feature, num_channel)

def feat_mav(x):    # mean absolute value
    num_batch, window_size, num_channel = x.shape
    y = np.mean(np.abs(x),axis=1)
    y = y.reshape(num_batch,1,num_channel)
    return y # shape: (num_batch, 1, num_channel)

def feat_mavs(x):   # Mean Absolute Value Slope
    num_batch, window_size, num_channel = x.shape
    x_mav = feat_mav(x) # shape: (num_batch, 1, num_channel)
    y = x_mav[:,:,1:] - x_mav[:,:,:-1]
    return y # shape: (num_batch, 1, num_channel-1)

def feat_zc(x, th=5e-4):     # number of zero crossings
    num_batch, window_size, num_channel = x.shape
    # Assuming num_feature is 1 as we are only calculating zero crossings
    y = np.zeros((num_batch, 1, num_channel))
    
    for i in range(num_batch):
        for j in range(num_channel):
            # Get the current window
            window = x[i, :, j] # shape: (window_size,)
            window_leftshift = np.concatenate((window[1:],window[-1].reshape(1)))
            isLargerTH = np.abs(window - window_leftshift) >= th

            window_sign = np.sign(window)   # shape: (window_size,)
            window_sign_rightshift = np.concatenate((window_sign[0].reshape(1),window_sign[0:-1]))

            isCross = (window_sign*window_sign_rightshift==-1)
            
            crossings = (isCross & isLargerTH).sum()

            y[i, 0, j] = crossings
    
    return y # shape: (num_batch, 1, num_channel)

def feat_ssc(x, th=5e-4):   # Slope Sign Changes
    num_batch, window_size, num_channel = x.shape
    # Assuming num_feature is 1 as we are only calculating zero crossings
    y = np.zeros((num_batch, 1, num_channel))

    for i in range(num_batch):
        for j in range(num_channel):
            # Get the current window
            window_1 = x[i, 0:-2, j] # shape: (window_size-2,)
            window_2 = x[i, 1:-1, j] # shape: (window_size-2,)
            window_3 = x[i, 2:  , j] # shape: (window_size-2,)

            isLargerTH1 = np.abs(window_2 - window_1) >= th
            isLargerTH2 = np.abs(window_2 - window_3) >= th
            isLargeTH = isLargerTH1 | isLargerTH2

            isMaximal = (window_2 > window_1) & (window_2 > window_3)
            isMinimal = (window_2 < window_1) & (window_2 < window_3)
            isLocal = isMaximal | isMinimal

            y[i, 0, j] = (isLocal & isLargeTH).sum()

    return y # shape: (num_batch, 1, num_channel)

def feat_wl(x):    # Waveform Length
    num_batch, window_size, num_channel = x.shape
    y = np.sum(np.abs(x[:,1:,:] - x[:,:-1,:]),axis=1)
    y = y.reshape(num_batch,1,num_channel)
    return y # shape: (num_batch, 1, num_channel)

def feat_Hudgins_FS(x):
    num_batch, window_size, num_channel = x.shape
    y1 = np.squeeze(feat_mav(x))
    y2 = np.squeeze(feat_mavs(x))
    y3 = np.squeeze(feat_zc(x))
    y4 = np.squeeze(feat_ssc(x))
    y5 = np.squeeze(feat_wl(x))
    y = np.concatenate((y1,y2,y3,y4,y5), axis=1) # shape: (num_batch, 5*num_channel-1)
    y = y.reshape(num_batch,1,y.shape[1])

    return y # shape: (num_batch, 1, 5*num_channel-1)