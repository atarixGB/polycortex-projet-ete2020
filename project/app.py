import mne
import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, signal
from scipy.integrate import simps
from pylsl import resolve_stream, stream_inlet

# OpenBCI Cyton Board
EPOCHS_DURATION = 3
BOARD_SAMPLING_FREQUENCY = 250
CHANNELS = ['Fz', 'FPz']

# ESP32
PORT = '/dev/tty.usbserial-0001'
BAUDRATE = 115200

def get_stream_inlet():
    print("looking for an EEG control stream...")
    streams = resolve_stream('type', 'EEG')
    return StreamInlet(streams[0])

def pull_eeg_data(inlet):
    data = []
    for _ in range(EPOCHS_DURATION * BOARD_SAMPLING_FREQUENCY):
        sample, _ = inlet.pull_sample() # use 'timestamp' instead of '_' to get timestamps
        data.append(sample)
    return data

def create_mne_epochs(epoch: np.array):
    epoch = np.delete(epoch, (-1), axis=0)
    info = mne.create_info(ch_names=CHANNELS, sfreq=BOARD_SAMPLING_FREQUENCY, ch_types=['eeg'] * len(CHANNELS), verbose=False)
    raw = mne.io.RawArray(epoch, info, verbose=False) # create MNE object from Numpy array
    return mne.make_fixed_length_epochs(raw, duration=EPOCHS_DURATION, preload=True, verbose=False)

def main():
    # pipeline = joblib.load('../saved_models/svm.joblib')
    stream_inlet = get_stream_inlet()
    ser = serial.Serial(PORT, BAUDRATE)

    while True:
        data = pull_eeg_data(stream_inlet)
        epochs = create_mne_epochs(data)
        # X = create_features_matrix(epochs)
        # Y = pipeline.predict(X)
        # ser.write(b'True') if Y else ser.write(b'False')

main()