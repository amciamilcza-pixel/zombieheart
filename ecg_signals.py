"""
ecg_signals.py
-----------------
ECG is simulated using numpy math fucntions, since the global mit database was not working T-T

The PQRST shape is modelled as a sum of Gaussians matching clinical ECG
morphology (P bump -> Q dip -> tall R spike -> S dip -> T bump -> flat).

Three heart types
-----------------
  normal      : 70 BPM — clean upright PQRST, stable
  bradycardia : 30 BPM — INVERTED around x-axis (everything flipped:
                P bump becomes negative, R spike points down, S dip
                points UP, T bump becomes negative), amplitude 45% - created 
                as a made up heartbead of a dying/decomposing zombie
  arrhythmia  : 140 BPM — random timing jitter ±35%, random amplitude
                per beat — chaotic zombie heart, turned not too long ago and agressive 

IMPORTANT — noise 
-----------------------------
All three signals are generated clean, then add_apocalypse_noise()
buries them in three layers of physical noise. The whole point of the
project is that the DFT filter then recovers/cleans up the signal from the
mess. So the raw input to the DFT pipeline is always the noisy version.

Noise layers:
  1. White noise  — electrical interference, amplitude set by SNR
  2. Baseline wander 0.2Hz — breathing/movement artefact
  3. 50 Hz power-line hum  — bunker generator interference (typical for EU)
"""

import numpy as np

FS       = 500
DURATION = 10.0


# ── PQRST template ──────────────────────────────────────────────────

def _pqrst(t, amp=1.0):
    """
    Single PQRST complex centred at t=0.

    Matches clinical ECG lead II morphology:
      P  wave : small positive bump    ~160ms before R  (atrial depolarisation)
      Q  wave : small negative dip     ~50ms before R   (septal depolarisation)
      R  peak : dominant positive spike at t=0           (ventricular depolarisation)
      S  wave : negative dip           ~55ms after R    (late ventricular depol.)
      T  wave : positive bump          ~220ms after R   (ventricular repolarisation)

    Amplitudes are proportional to amp. Inverting amp (pass -amp) flips
    everything around the x-axis, giving the bradycardia 'dying' pattern.
    """
    p  =  amp * 0.15 * np.exp(-0.5 * ((t + 0.250) / 0.040) ** 2)
    q  =  amp *-0.10 * np.exp(-0.5 * ((t + 0.050) / 0.018) ** 2)
    r  =  amp * 1.00 * np.exp(-0.5 * (t             / 0.022) ** 2)
    s  =  amp *-0.25 * np.exp(-0.5 * ((t - 0.055) / 0.020) ** 2)
    tw =  amp * 0.20 * np.exp(-0.5 * ((t - 0.220) / 0.055) ** 2)
    return p + q + r + s + tw


def _build_ecg(fs, duration, beat_times, amplitudes):
    """Place PQRST templates at each beat time and sum onto a zero array."""
    N   = int(fs * duration)
    t   = np.arange(N) / fs
    ecg = np.zeros(N)
    for tb, amp in zip(beat_times, amplitudes):
        local = t - tb
        mask  = np.abs(local) < 0.55   # only update samples within 550ms of beat
        ecg[mask] += _pqrst(local[mask], amp=amp)
    return t, ecg


# ── public generators ─────────────────────────────────────────────────────────

def generate_normal(fs=FS, duration=DURATION, seed=0):
    """
    Healthy survivor heart.
    70 BPM, perfectly regular, upright PQRST, amplitude = 1.0.
    """
    period     = 60.0 / 70.0
    beat_times = np.arange(period, duration - 0.4, period)
    amplitudes = np.ones(len(beat_times))
    return _build_ecg(fs, duration, beat_times, amplitudes)


def generate_bradycardia(fs=FS, duration=DURATION, seed=1):
    """
    Dying / inverse heart.
    30 BPM, signal FULLY INVERTED around x-axis, amplitude scaled to 45%.

    Inversion means:
      - P bump  → negative dip
      - R spike → points DOWN (big negative spike)
      - S dip   → points UP
      - T bump  → negative dip

    This is achieved simply by passing a negative amplitude to _pqrst(),
    which flips every Gaussian component simultaneously.

    Physiological note: this pattern is real — it appears in posterior
    myocardial infarction and certain bundle-branch blocks where the
    electrical axis of depolarisation is reversed relative to the lead.
    """
    period     = 60.0 / 30.0
    beat_times = np.arange(period, duration - 0.4, period)
    # Negative amplitude = full x-axis inversion; 0.45 = weak dying heart
    amplitudes = -0.45 * np.ones(len(beat_times))
    return _build_ecg(fs, duration, beat_times, amplitudes)


def generate_arrhythmia(fs=FS, duration=DURATION, seed=2):
    """
    Zombie heart — fast and completely chaotic.
    ~140 BPM average with:
      - ±35% random beat-to-beat timing jitter  (no consistent period)
      - random amplitude per beat in [0.5, 1.6] (erratic pumping strength)

    The lack of a consistent period is exactly why the GLOBAL DFT gives
    a misleading result (stress test 3B): it smears all the irregular
    inter-beat intervals into a broad hump rather than a clean spike.
    Only the sliding-window DFT (STFT) reveals the true non-stationarity.
    """
    rng     = np.random.default_rng(seed)
    nominal = 60.0 / 140.0   # ~0.429 s

    beat_times = []
    cursor     = nominal
    while cursor < duration - 0.5:
        beat_times.append(cursor)
        jitter  = rng.uniform(-0.35, 0.35) * nominal
        cursor += nominal + jitter
    beat_times = np.array(beat_times)

    amplitudes = rng.uniform(0.5, 1.6, size=len(beat_times))
    return _build_ecg(fs, duration, beat_times, amplitudes)


# ── noise ─────────────────────────────────────────────────────────────────────

def add_apocalypse_noise(ecg, fs=FS, snr_db=5, seed=99):
    """
    Bury the clean ECG in three layers of realistic noise.

    This is the STARTING POINT for the DFT analysis — the project
    demonstrates recovering the signal from this mess.

    Layer 1 — White Gaussian noise
        The most important layer. Amplitude is computed so that the
        signal-to-noise ratio equals snr_db (amplitude dB definition:
        SNR_dB = 20*log10(signal_rms / noise_rms)).
        At SNR=5 dB the ECG is almost invisible in the time domain.
        Completely random — new noise every call (unless seed fixed).

    Layer 2 — Baseline wander at 0.2 Hz
        A slow sine wave with random phase, amplitude = 30% of signal RMS.
        In real ECGs this comes from breathing moving the electrode.

    Layer 3 — 50 Hz power-line hum
        A sine at exactly 50 Hz (European grid frequency) with random phase,
        amplitude = 25% of signal RMS.
        The DFT notch filter removes this precisely because we know its
        exact frequency — this is a key advantage of frequency-domain filtering.

    Parameters
    ----------
    ecg    : clean ECG signal (1-D array)
    fs     : sample rate in Hz
    snr_db : signal-to-noise ratio in dB (5 = buried, 20 = mild noise)
    seed   : integer seed for reproducibility (change for different noise)
    """
    rng     = np.random.default_rng(seed)
    N       = len(ecg)
    t       = np.arange(N) / fs
    sig_rms = np.sqrt(np.mean(ecg ** 2)) + 1e-10

    # Layer 1: white Gaussian noise scaled to target SNR
    noise_rms = sig_rms / (10 ** (snr_db / 20.0))
    white     = rng.normal(0, noise_rms, N)

    # Layer 2: baseline wander
    wander = 0.30 * sig_rms * np.sin(2*np.pi * 0.20 * t + rng.uniform(0, 2*np.pi))

    # Layer 3: 50 Hz hum
    hum    = 0.25 * sig_rms * np.sin(2*np.pi * 50.0 * t + rng.uniform(0, 2*np.pi))

    return ecg + white + wander + hum
