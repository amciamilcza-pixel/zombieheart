"""
run_me.py
=========
ZOMBIE APOCALYPSE ECG -- DFT Signals & Systems Project
Signals and Systems 4CA20, 2025-2026
------------------------------------------------------
Run this file to reproduce ALL figures, audio files, and results:

    python run_me.py

Outputs are saved to ./figures/
  fig1_three_hearts.png          -- three cardiac signatures (time + DFT)
  fig2_recovery_normal.png       -- DFT noise recovery, normal heart
  fig2_recovery_bradycardia.png  -- DFT noise recovery, bradycardia
  fig2_recovery_arrhythmia.png   -- DFT noise recovery, arrhythmia
  fig3a_window_spectra.png       -- parameter sensitivity: window spectra
  fig3b_window_accuracy.png      -- parameter sensitivity: BPM accuracy vs N
  fig4_noise_robustness.png      -- noise robustness: BPM error vs SNR
  fig5_failure_cases.png         -- failure cases: leakage + non-stationarity
  
  *NEW AUDIO OUTPUTS:*
  audio_normal_clean.wav / audio_normal_noisy.wav / audio_normal_recovered.wav
  audio_bradycardia_clean.wav / audio_bradycardia_noisy.wav / audio_bradycardia_recovered.wav
  audio_arrhythmia_clean.wav / audio_arrhythmia_noisy.wav / audio_arrhythmia_recovered.wav
"""

import os
import sys
import time
import numpy as np
from scipy.io import wavfile

print("=" * 60)
print("  ☣  ZOMBIE ECG PROJECT -- generating all figures & audio")
print("=" * 60)

# ── imports ------------------------------------------------------────────────
from ecg_signals  import (generate_normal, generate_bradycardia,
                           generate_arrhythmia, add_apocalypse_noise)
from dft_filter   import recover_ecg, heart_rate_from_dft
from stress_tests import (test_window_sensitivity, test_noise_robustness,
                           test_failure_cases)
from plots        import (plot_three_hearts, plot_recovery_pipeline,
                           plot_window_sensitivity, plot_noise_robustness,
                           plot_failure_cases)

FS       = 500     # sample rate (Hz) -- standard clinical ECG
DURATION = 10.0    # seconds
SNR_DB   = 5       # dB -- pretty buried in noise (apocalypse is harsh)


# ── sonification function ------------------------------------------------────
def sonify_heartbeats(t, ecg_signal, fs, filename="heart_sound.wav"):
    """
    Sonifies an ECG signal using Frequency Modulation (FM Synthesis).
    Maps the low-frequency ECG voltage directly to an audible pitch.
    """
    audio_fs = 44100  # Standard CD audio rate
    
    # Interpolate the low-frequency ECG timeline up to the audio timeline
    t_audio = np.arange(0, t[-1], 1.0 / audio_fs)
    ecg_audio = np.interp(t_audio, t, ecg_signal)
    
    # Normalize the ECG wave between 0.0 and 1.0 safely
    min_val = np.min(ecg_audio)
    max_val = np.max(ecg_audio)
    ecg_norm = (ecg_audio - min_val) / (max_val - min_val + 1e-6)
    
    # FM Synthesis parameters
    f_carrier = 440.0     # Base frequency (Hz)
    f_mod_depth = 440.0   # How high up the frequency climbs during an R-spike
    
    # Integrate frequency changes over time to yield continuous tracking phase
    phase = 2 * np.pi * (f_carrier * t_audio + f_mod_depth * np.cumsum(ecg_norm) / audio_fs)
    audio_wave = np.sin(phase)
    
    # Scale to 16-bit PCM Signed Integer Audio
    audio_pcm = np.int16(audio_wave * 32767)
    
    os.makedirs("./figures", exist_ok=True)
    wavfile.write(f"./figures/{filename}", audio_fs, audio_pcm)
    print(f"      ✓ Audio sonification saved: ./figures/{filename}")


# ------------------------------------------------------------------------─────
# SECTION 1 -- Generate the three heart signals & Sonify Clean Baselines
# ------------------------------------------------------------------------─────
print("\n[1/5] Generating cardiac signals & saving clean audio baseline…")

t, ecg_normal       = generate_normal     (fs=FS, duration=DURATION)
_, ecg_bradycardia  = generate_bradycardia(fs=FS, duration=DURATION)
_, ecg_arrhythmia   = generate_arrhythmia (fs=FS, duration=DURATION)

signals_clean = {
    "normal":      ecg_normal,
    "bradycardia": ecg_bradycardia,
    "arrhythmia":  ecg_arrhythmia,
}

# Print true BPM for reference
print(f"   Normal       BPM (DFT estimate): {heart_rate_from_dft(ecg_normal, FS):.1f}")
sonify_heartbeats(t, ecg_normal, FS, "audio_normal_clean.wav")

print(f"   Bradycardia  BPM (DFT estimate): {heart_rate_from_dft(ecg_bradycardia, FS):.1f}")
sonify_heartbeats(t, ecg_bradycardia, FS, "audio_bradycardia_clean.wav")

print(f"   Arrhythmia   BPM (DFT estimate): {heart_rate_from_dft(ecg_arrhythmia, FS):.1f}")
sonify_heartbeats(t, ecg_arrhythmia, FS, "audio_arrhythmia_clean.wav")

# Figure 1
plot_three_hearts(t, signals_clean, FS)

# ------------------------------------------------------------------------─────
# SECTION 2 -- Bury in noise → DFT recovery pipeline & Sonify Noisy + Recovered
# ------------------------------------------------------------------------─────
print("\n[2/5] Running DFT recovery pipeline & saving noisy/recovered audio…")

configs = [
    ("normal",      ecg_normal,      "normal"),
    ("bradycardia", ecg_bradycardia,  "bradycardia"),
    ("arrhythmia",  ecg_arrhythmia,  "arrhythmia"),
]

for label, ecg_clean, color_key in configs:
    noisy = add_apocalypse_noise(ecg_clean, FS, snr_db=SNR_DB)
    recovered, X_noisy, X_recovered, freqs = recover_ecg(noisy, FS)

    bpm_noisy     = heart_rate_from_dft(noisy, FS)
    bpm_recovered = heart_rate_from_dft(recovered, FS)
    print(f"   {label:15s}  BPM from noisy={bpm_noisy:.1f}  BPM after DFT filter={bpm_recovered:.1f}")

    # Sonify the noisy version
    sonify_heartbeats(t, noisy, FS, f"audio_{label}_noisy.wav")
    
    # Sonify the recovered/filtered version
    sonify_heartbeats(t, recovered, FS, f"audio_{label}_recovered.wav")

    plot_recovery_pipeline(t, ecg_clean, noisy, recovered,
                           X_noisy, X_recovered, freqs, FS,
                           label=label, color_key=color_key)

# ------------------------------------------------------------------------─────
# SECTION 3 -- Stress test 1: window / parameter sensitivity
# ------------------------------------------------------------------------─────
print("\n[3/5] Stress test 1 -- parameter sensitivity (window length) …")
t0 = time.time()
ws_data = test_window_sensitivity(fs=FS, duration=DURATION)
plot_window_sensitivity(ws_data)

print(f"   Window sizes tested : {ws_data['window_sizes']}")
print(f"   Df range            : {min(ws_data['delta_f']):.3f} – "
      f"{max(ws_data['delta_f']):.3f} Hz")
print(f"   BPM estimates       : {[f'{b:.1f}' for b in ws_data['hr_estimates']]}")
print(f"   ({time.time()-t0:.1f}s)")

# ------------------------------------------------------------------------─────
# SECTION 4 -- Stress test 2: noise robustness
# ------------------------------------------------------------------------─────
print("\n[4/5] Stress test 2 -- noise robustness (SNR sweep) …  ", end="", flush=True)
t0 = time.time()
nr_data = test_noise_robustness(fs=FS, duration=DURATION)
plot_noise_robustness(nr_data)
print(f"done  ({time.time()-t0:.1f}s)")

# Print the SNR threshold where error exceeds 10 BPM for each type
for name, d in nr_data.items():
    errors = np.array(d["hr_error"])
    snrs   = np.array(d["snr_levels"])
    bad_snrs = snrs[errors > 10]
    if len(bad_snrs):
        print(f"   {name:15s}: DFT detection breaks below "
              f"SNR ≈ {bad_snrs.max():.1f} dB")

# ------------------------------------------------------------------------─────
# SECTION 5 -- Stress test 3: failure / limitation cases
# ------------------------------------------------------------------------─────
print("\n[5/5] Stress test 3 -- failure cases (leakage + non-stationarity) …")
fc_data = test_failure_cases(fs=FS)
plot_failure_cases(fc_data)

# ------------------------------------------------------------------------─────
# SUMMARY
# ------------------------------------------------------------------------─────
print("\n" + "=" * 60)
print("  ✓  All figures and audio files saved to ./figures/")
print("=" * 60)