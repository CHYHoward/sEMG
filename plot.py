import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

def plot_FFT(x, Fs):
    # Perform the FFT
    n = len(x)  # Length of signal
    freq_axis = np.arange(0, (Fs/2), Fs/n) # Create the frequency vector

    # Plot the FFT
    plt.figure(figsize=(12, 4))
    for i in range(x.shape[1]):
        X = np.fft.fft(x[:,i])/n  # FFT normalized by the number of samples
        X = X[range(int(n/2))]  # Remove the symmetric part of the FFT
        plt.plot(freq_axis, abs(X))

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

def plot_cm(y_gold, y_pred, log_name=None):

    # Calculate the confusion matrix
    cm = confusion_matrix(y_gold, y_pred, normalize='true')*100

    # Plot the confusion matrix using seaborn and matplotlib
    plt.figure(figsize=(12, 12))
    sns.set(font_scale=0.6)
    sns.heatmap(cm, annot=True, fmt="3.0f", cbar=False, square=True, xticklabels=True, yticklabels=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    if log_name is not None:
        # Save as npy and png
        PATH = f"Results/{log_name}"
        if not os.path.isdir(PATH):
            os.mkdir(PATH)

        plt.savefig(f"{PATH}/confusion_matrix.svg")
        plt.savefig(f"{PATH}/confusion_matrix.png", dpi=300)
        np.save(f"{PATH}/y_pred.npy", y_pred)
        np.save(f"{PATH}/y_gold.npy", y_gold)