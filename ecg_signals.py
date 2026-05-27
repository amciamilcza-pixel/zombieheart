"""
ecg_signals.py
--------------
Synthetic ECG signal generator for the Zombie Apocalypse DFT project.

Three heart types:
  - Normal   : healthy sinus rhythm ~70 BPM
  - Bradycardia ("Inverse / dying") : very slow ~30 BPM, weak amplitude
  - Arrhythmia ("Zombie / messed up"): fast + irregular ~140 BPM with
    random timing jitter and amplitude variation

All signals are buried in apocalyptic background noise and must be
recovered using DFT-based filtering.
"""

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _pqrst_template(t, amplitude=1.0):
    """
    Single PQRST complex centred at t=0 using a sum of Gaussians.
    Returns a 1-D array evaluated on the supplied time vector.
    """
    # R-peak (dominant spike)
    r  = amplitude * 1.20 * np.exp(-0.5 * (t / 0.03) ** 2)
    # Q / S flanks
    q  = amplitude * -0.15 * np.exp(-0.5 * ((t + 0.04) / 0.015) ** 2)
    s  = amplitude * -0.20 * np.exp(-0.5 * ((t - 0.04) / 0.015) ** 2)
    # P wave
    p  = amplitude *  0.25 * np.exp(-0.5 * ((t + 0.20) / 0.035) ** 2)
    # T wave
    tw = amplitude *  0.35 * np.exp(-0.5 * ((t - 0.20) / 0.060) ** 2)
    return r + q + s + p + tw


def _build_ecg(fs, duration, bpm, beat_times=None,
               amplitude_fn=None, rng=None):
    """
    Place PQRST templates on a timeline.

    Parameters
    ----------
    fs          : sample rate (Hz)
    duration    : signal length (s)
    bpm         : nominal beats per minute
    beat_times  : override automatic placement (array of seconds)
    amplitude_fn: callable(n_beats) → amplitude per beat, or None for const 1
    rng         : numpy.random.Generator for reproducibility
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = int(fs * duration)
    t = np.arange(N) / fs
    ecg = np.zeros(N)

    if beat_times is None:
        period = 60.0 / bpm
        beat_times = np.arange(period, duration - 0.3, period)

    n_beats = len(beat_times)
    amplitudes = amplitude_fn(n_beats) if amplitude_fn else np.ones(n_beats)

    for tb, amp in zip(beat_times, amplitudes):
        # local time axis around this beat
        local_t = t - tb
        mask = np.abs(local_t) < 0.5   # only update nearby samples
        ecg[mask] += _pqrst_template(local_t[mask], amplitude=amp)

    return t, ecg


# ── public API ────────────────────────────────────────────────────────────────

def generate_normal(fs=500, duration=10.0, seed=0):
    """Healthy sinus rhythm ~70 BPM, stable amplitude."""
    rng = np.random.default_rng(seed)
    t, ecg = _build_ecg(fs, duration, bpm=70, rng=rng)
    return t, ecg


def generate_bradycardia(fs=500, duration=10.0, seed=1):
    """
    Slow, weak heartbeat (~30 BPM, reduced amplitude).
    Represents the 'dying / inverse' zombie heart.
    """
    rng = np.random.default_rng(seed)
    t, ecg = _build_ecg(
        fs, duration, bpm=30,
        amplitude_fn=lambda n: 0.45 * np.ones(n),
        rng=rng
    )
    return t, ecg


def generate_arrhythmia(fs=500, duration=10.0, seed=2):
    """
    Fast, irregular heartbeat (~140 BPM average) with:
      - random inter-beat timing jitter  ±30 % of nominal period
      - random beat-to-beat amplitude variation  0.6 – 1.4×
    Represents the fully turned zombie heart.
    """
    rng = np.random.default_rng(seed)
    nominal_period = 60.0 / 140.0          # ~0.43 s

    # Build irregular beat times
    beat_times = []
    t_cursor = nominal_period
    while t_cursor < duration - 0.3:
        beat_times.append(t_cursor)
        jitter = rng.uniform(-0.30, 0.30) * nominal_period
        t_cursor += nominal_period + jitter
    beat_times = np.array(beat_times)

    amp_fn = lambda n: rng.uniform(0.6, 1.4, size=n)
    t, ecg = _build_ecg(fs, duration, bpm=140,
                        beat_times=beat_times,
                        amplitude_fn=amp_fn, rng=rng)
    return t, ecg


def add_apocalypse_noise(ecg, fs, snr_db=5, seed=99):
    """
    Bury the ECG in realistic 'apocalypse' noise:
      - Gaussian white noise   (electrical interference)
      - Low-frequency drift    (movement / breathing artefact)
      - 50 Hz power-line hum   (generator noise in the bunker)

    snr_db : signal-to-noise ratio in dB (lower = harder to recover)
    """
    rng = np.random.default_rng(seed)
    N = len(ecg)
    t = np.arange(N) / fs

    # White noise
    sig_power = np.mean(ecg ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    white = rng.normal(0, np.sqrt(noise_power), N)

    # Baseline wander (0.2 Hz)
    wander = 0.3 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 2*np.pi))

    # 50 Hz power-line hum
    hum = 0.25 * np.sin(2 * np.pi * 50 * t)

    return ecg + white + wander + hum
