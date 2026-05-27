"""
dft_filter.py
-------------
DFT-based signal recovery for the Zombie Apocalypse ECG project.

All filtering is done EXPLICITLY in the frequency domain using numpy's FFT
so the DFT is the core analytical tool, not just a visualisation step.

Functions
---------
dft_bandpass       : zero out DFT bins outside [f_low, f_high]
dft_notch          : zero out DFT bins around a specific frequency (e.g. 50 Hz hum)
recover_ecg        : full pipeline: bandpass + notch → inverse DFT
spectral_magnitude : returns frequency axis + magnitude spectrum
dominant_frequency : finds the strongest frequency peak in a given band
heart_rate_from_dft: estimates BPM from the DFT peak in the cardiac band
"""

import numpy as np


def spectral_magnitude(signal, fs):
    """
    Compute one-sided magnitude spectrum via DFT.

    Returns
    -------
    freqs : frequency axis (Hz)
    mag   : magnitude (same units as signal)
    X     : full complex DFT coefficients (needed for inverse)
    """
    N = len(signal)
    X = np.fft.fft(signal)               # DFT
    freqs = np.fft.fftfreq(N, d=1.0/fs)  # frequency axis
    # one-sided (positive frequencies only for display)
    half = N // 2
    return freqs[:half], np.abs(X[:half]), X


def dft_bandpass(signal, fs, f_low, f_high):
    """
    Ideal bandpass filter implemented in the DFT domain.

    Steps
    -----
    1. Compute DFT of signal
    2. Zero all bins outside [f_low, f_high]  (both positive and negative)
    3. Inverse DFT → filtered signal (real part only)

    This is the fundamental DFT filtering operation.
    """
    N = len(signal)
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1.0/fs)

    # Build mask: keep only bins inside the passband
    mask = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)
    X_filtered = X * mask                # zero out everything outside band

    # Inverse DFT → back to time domain
    filtered = np.fft.ifft(X_filtered).real
    return filtered, X_filtered, freqs


def dft_notch(signal, fs, f_notch, bandwidth=2.0):
    """
    Notch filter: remove a narrow band around f_notch (e.g. 50 Hz hum).
    Implemented in DFT domain — zeros out bins within ±bandwidth/2 of f_notch.
    """
    N = len(signal)
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1.0/fs)

    # Kill bins near ±f_notch
    notch_mask = np.abs(np.abs(freqs) - f_notch) < bandwidth / 2
    X[notch_mask] = 0

    filtered = np.fft.ifft(X).real
    return filtered


def recover_ecg(noisy_signal, fs, f_low=0.5, f_high=45.0, notch_hz=50.0):
    """
    Full DFT-based ECG recovery pipeline:
      1. Notch filter  → remove 50 Hz power-line hum
      2. Bandpass      → keep only cardiac frequency content (0.5–45 Hz)
      3. Inverse DFT   → recovered time-domain ECG

    Parameters
    ----------
    noisy_signal : contaminated signal
    fs           : sample rate (Hz)
    f_low        : lower edge of cardiac bandpass (Hz)
    f_high       : upper edge of cardiac bandpass (Hz)
    notch_hz     : power-line frequency to notch out (Hz)

    Returns
    -------
    recovered    : clean ECG estimate
    X_noisy      : DFT of noisy input  (for plotting)
    X_recovered  : DFT after filtering (for plotting)
    freqs        : frequency axis
    """
    # Step 1: notch out the hum
    after_notch = dft_notch(noisy_signal, fs, f_notch=notch_hz, bandwidth=2.0)

    # Step 2 + 3: bandpass + inverse DFT
    recovered, X_recovered, freqs = dft_bandpass(after_notch, fs, f_low, f_high)

    # Also get DFT of original noisy signal for comparison plots
    X_noisy = np.fft.fft(noisy_signal)

    return recovered, X_noisy, X_recovered, freqs


def heart_rate_from_dft(signal, fs, search_band=(0.5, 3.5)):
    """
    Estimate heart rate (BPM) from the dominant DFT peak in the
    cardiac fundamental band (default 0.5–3.5 Hz = 30–210 BPM).

    Returns estimated BPM (float).
    """
    N = len(signal)
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1.0/fs)

    half = N // 2
    freqs_pos = freqs[:half]
    mag_pos = np.abs(X[:half])

    # Restrict to search band
    mask = (freqs_pos >= search_band[0]) & (freqs_pos <= search_band[1])
    if not np.any(mask):
        return 0.0

    peak_freq = freqs_pos[mask][np.argmax(mag_pos[mask])]
    return peak_freq * 60.0  # Hz → BPM
