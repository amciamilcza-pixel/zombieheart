"""
dft_filter.py

# FFT-based ECG filtering.
# Removes baseline drift and power-line noise in frequency domain.


Bandpass filter using FFT masking.
Keeps only frequencies inside the desired range.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt, iirnotch


def spectral_magnitude(signal, fs):
    """
    One-sided DFT magnitude spectrum.

    Returns
    -------
    freqs : frequency axis Hz (positive only)
    mag   : magnitude spectrum
    X     : full complex DFT (needed for inverse transform)
    """
    N = len(signal)
    X     = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    half  = N // 2
    return freqs[:half], np.abs(X[:half]), X


def dft_bandpass(signal, fs, f_low, f_high):
    """
    Ideal rectangular bandpass filter in the DFT domain.

    Steps
    -----
    1. Compute DFT  X[k] = sum_n x[n] * e^{-j2pi*k*n/N}
    2. Zero all bins k where |f[k]| < f_low  or  |f[k]| > f_high
    3. Inverse DFT  x_filt[n] = (1/N) * sum_k X_filt[k] * e^{j2pi*k*n/N}

    This is the fundamental DFT filtering operation — we operate
    directly on the spectrum, not with a convolution kernel.
    """
    N      = len(signal)
    X      = np.fft.fft(signal)
    freqs  = np.fft.fftfreq(N, d=1.0 / fs)

    # Passband mask (both positive and negative frequencies)
    mask       = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)
    X_filtered = X * mask

    filtered = np.fft.ifft(X_filtered).real
    return filtered, X_filtered, freqs


def dft_notch(signal, fs, f_notch, bandwidth=2.0):
    """
    Notch filter in DFT domain: zero bins within ±bandwidth/2 of f_notch.
    Used to remove 50 Hz power-line hum before bandpass filtering.
    """
    N      = len(signal)
    X      = np.fft.fft(signal)
    freqs  = np.fft.fftfreq(N, d=1.0 / fs)

    notch_mask    = np.abs(np.abs(freqs) - f_notch) < bandwidth / 2
    X[notch_mask] = 0

    return np.fft.ifft(X).real


def spectral_magnitude(signal, fs):
    signal = np.asarray(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1/fs)
    fft_vals = np.fft.rfft(signal)
    mag = np.abs(fft_vals)
    return freqs, mag, fft_vals


def recover_ecg(signal, fs):
    """
    Clean a real ECG signal:
    1. remove mean (DC offset)
    2. notch filter at 50 Hz
    3. bandpass filter for ECG range
    """

    x = np.asarray(signal, dtype=float)

    # Remove baseline offset so raw and filtered can be compared fairly
    x_centered = x - np.mean(x)

    # 50 Hz notch filter
    b_notch, a_notch = iirnotch(w0=50, Q=30, fs=fs)
    x_notched = filtfilt(b_notch, a_notch, x_centered)

    # Stronger ECG bandpass
    # 0.5 Hz removes slow drift
    # 20 Hz removes a lot of high-frequency noise
    b_band, a_band = butter(4, [0.5, 20], btype='bandpass', fs=fs)
    x_filtered = filtfilt(b_band, a_band, x_notched)

    # Spectra for plotting
    freqs, mag_before, _ = spectral_magnitude(x_centered, fs)
    _, mag_after, _ = spectral_magnitude(x_filtered, fs)

    return x_filtered, freqs, mag_before, mag_after


def heart_rate_from_dft(signal, fs):
    """
    Estimate BPM from the dominant low-frequency peak in the ECG spectrum.
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)

    freqs = np.fft.rfftfreq(len(signal), d=1/fs)
    mag = np.abs(np.fft.rfft(signal))

    # Search only plausible heart-rate frequencies:
    # 0.6 Hz to 3.5 Hz -> 36 BPM to 210 BPM
    mask = (freqs >= 0.6) & (freqs <= 3.5)

    freqs_hr = freqs[mask]
    mag_hr = mag[mask]

    peaks, _ = find_peaks(mag_hr)

    if len(peaks) == 0:
        dominant_freq = freqs_hr[np.argmax(mag_hr)]
    else:
        dominant_peak = peaks[np.argmax(mag_hr[peaks])]
        dominant_freq = freqs_hr[dominant_peak]

    bpm = dominant_freq * 60
    return bpm


def heart_rate_from_dft(signal, fs, search_band=(0.4, 3.5)):
    """
    Estimate heart rate (BPM) from the DFT by finding the FUNDAMENTAL
    frequency — the lowest significant spectral peak in the cardiac band.

    Why lowest peak, not highest?
    The ECGSYN model (used by NeuroKit2) generates strong harmonics.
    The 2nd harmonic (2× fundamental) can have higher magnitude than the
    fundamental itself. We want the FUNDAMENTAL (f0) because:
        f0 [Hz] × 60 = heart rate [BPM]

    Strategy: find all peaks above 30% of band maximum, return lowest.
    This correctly identifies f0 even when harmonics are stronger.

    Returns BPM (float), or 0 if no peak found.
    """
    N       = len(signal)
    X       = np.fft.fft(signal)
    freqs   = np.fft.fftfreq(N, d=1.0 / fs)
    half    = N // 2
    f_pos   = freqs[:half]
    mag_pos = np.abs(X[:half])

    # Restrict to search band
    mask    = (f_pos >= search_band[0]) & (f_pos <= search_band[1])
    if not np.any(mask):
        return 0.0

    f_band = f_pos[mask]
    m_band = mag_pos[mask]

    # Minimum spacing between peaks: 0.2 Hz (avoids noise spikes)
    min_dist = max(1, int(0.2 / (f_pos[1] - f_pos[0])))
    threshold = 0.30 * m_band.max()

    peaks, _ = find_peaks(m_band, height=threshold, distance=min_dist)
    if len(peaks) == 0:
        # Fallback: raw argmax
        return f_band[np.argmax(m_band)] * 60.0

    # Return lowest-frequency significant peak = fundamental
    return f_band[peaks[0]] * 60.0
