# %%
import numpy as np 
import matplotlib.pyplot as plt

# Save/Load as mat.
from scipy.io import savemat, loadmat
import h5py
import mat73 # https://github.com/skjerns/mat7.3

# Other
import argparse
import multiprocessing as mp
import os

def mat2np_segment(mat, idx_subject, idx_exercise):
    subject = np.squeeze(mat['subject'])                # subject no., value: 1~40, shape: 1x1
    emg = mat['emg']                                    # emg signal, shape: ((num_trial*(5+3)*fs)*num_gesture) * num_channel
    repetition = np.squeeze(mat['repetition'])          # repetition (i.e., trial), value: 0~6 shape: ((num_trial*(5+3)*fs)*num_gesture) * 1 
    restimulus = np.squeeze(mat['restimulus'])          # Gesture label, value: 0~49 shape: ((num_trial*(5+3)*fs)*num_gesture) * 1 

    num_channel = emg.shape[1]
    num_sample = emg.shape[0]
    num_trial = 6
    fs = 2000 # sampling frequency (sps)

    for idx_trial in range(1,num_trial+1):
        idx_repetition_trial = np.where(repetition == idx_trial)[0]
        repetition_trial = repetition[idx_repetition_trial]
        # restimulus_trial = restimulus[idx_repetition_trial]
        restimulus_trial = np.take(restimulus, idx_repetition_trial, mode='clip')
        emg_trial = emg[idx_repetition_trial,:]
        for idx_gesture in np.unique(restimulus_trial):
            if idx_gesture:   # Not the rest gesture (e.g., j != 0)
                idx_emg_trial_gesture = np.where(restimulus_trial == idx_gesture)[0]
                emg_trial_gesture = emg_trial[idx_emg_trial_gesture,:]
                PATH_seg_np = "Dataset/DB2/DB2_np/DB2_s_{}/exercise_{}/trial_{}".format(idx_subject,idx_exercise,idx_trial)
                # if the demo_folder directory is not present then create it. 
                if not os.path.exists(PATH_seg_np): 
                    os.makedirs(PATH_seg_np) 
                np.save("{}/G_{}".format(PATH_seg_np,idx_gesture), emg_trial_gesture)
                print("{}/G_{}".format(PATH_seg_np,idx_gesture))


def mat2np_segment_all_subject(subject_list=[i+1 for i in range(40)], exercise_list=[1,2,3]):
    for idx_subject in subject_list:
        for idx_exercise in exercise_list:
            FILE = "Dataset/DB2/DB2_s{}/S{}_E{}_A1.mat".format(idx_subject,idx_subject,idx_exercise)
            print("Opening  {} ...".format(FILE))
            mat = loadmat(FILE)
            mat2np_segment(mat, idx_subject, idx_exercise)


if __name__ == "__main__":
    mat2np_segment_all_subject()



