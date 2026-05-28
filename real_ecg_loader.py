import wfdb
import numpy as np


def load_real_ecg(record_name, start_sec=0, duration_sec=10, channel=0):
    """
    Loads a real ECG segment from the MIT-BIH Arrhythmia Database.

    record_name examples:
    "100" = normal-ish ECG
    "209" = fast rhythm section
    "232" = slow bradycardia ECG
    """

    # Read the header first, so we know the sampling frequency.
    header = wfdb.rdheader(record_name, pn_dir="mitdb")
    fs = int(header.fs)

    # Convert seconds into sample numbers.
    sampfrom = int(start_sec * fs)
    sampto = int((start_sec + duration_sec) * fs)

    # Read the actual ECG signal from PhysioNet.
    signals, fields = wfdb.rdsamp(
        record_name,
        pn_dir="mitdb",
        sampfrom=sampfrom,
        sampto=sampto,
        channels=[channel]
    )

    # signals is a 2D array. We only selected one channel, so take column 0.
    ecg = signals[:, 0]

    # Make a matching time axis.
    t = np.arange(len(ecg)) / fs

    return t, ecg, fs


def invert_ecg(ecg):
    """
    Makes the ECG upside down.
    This is your zombie/dying heart visual effect.
    """
    return -ecg