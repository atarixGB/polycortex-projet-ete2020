import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics 
from scipy import fftpack, signal
from scipy.integrate import simps

CYTON_BOARD_SAMPLING_RATE = 250.0
NYQUIST = CYTON_BOARD_SAMPLING_RATE/2

CHANNELS_NAME = ['index', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5','ch6', 'ch7', 'ch8', 'aux1', 'aux2', 'aux3', 'aux4', 'timestamp']

def read_eeg_data(openbci_file, channels_to_drop = []):
    eeg_data = pd.read_csv(openbci_file, skiprows=6)
    eeg_data.columns = CHANNELS_NAME

    eeg_dataframe = pd.DataFrame(eeg_data)

    if (len(channels_to_drop) > 0):
        eeg_dataframe.drop(channels_to_drop, axis=1, inplace=True)

    # Set time reference to 0 ms
    eeg_dataframe['timestamp'] = [float(data) - float(eeg_dataframe['timestamp'][0]) for data in eeg_dataframe['timestamp']]

    # Set time in seconds
    eeg_dataframe['timestamp'] = np.divide(eeg_dataframe['timestamp'], 1000)
    return eeg_dataframe
