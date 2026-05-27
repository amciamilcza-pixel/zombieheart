import os
from scipy.io import wavfile

def sonify_heartbeats(t, ecg_signal, fs, filename="heart_sound.wav"):
    """
    Sonifies an ECG signal using Frequency Modulation (FM Synthesis).
    Maps the low-frequency ECG voltage directly to an audible pitch.
    """
    # 1. Establish an audible audio sample rate (standard 44.1 kHz)
    audio_fs = 44100
    
    # 2. Interpolate the low-frequency ECG timeline up to the audio timeline
    t_audio = np.arange(0, t[-1], 1.0 / audio_fs)
    ecg_audio = np.interp(t_audio, t, ecg_signal)
    
    # 3. Normalize the ECG wave between 0.0 and 1.0
    ecg_norm = (ecg_audio - np.min(ecg_audio)) / (np.max(ecg_audio) - np.min(ecg_audio) + 1e-6)
    
    # 4. FM Synthesis: Map normalized voltage to an audible frequency band
    # Baseline normal skin potential = 440 Hz (A4 note). R-peak spike = 880 Hz (A5 note).
    f_carrier = 440.0
    f_mod_depth = 440.0
    
    # Integrate frequency to calculate instantaneous phase
    phase = 2 * np.pi * (f_carrier * t_audio + f_mod_depth * np.cumsum(ecg_norm) / audio_fs)
    audio_wave = np.sin(phase)
    
    # 5. Scale to 16-bit PCM Audio integers
    audio_pcm = np.int16(audio_wave * 32767)
    
    # Ensure target folder exists and save file
    os.makedirs("./figures", exist_ok=True)
    wavfile.write(f"./figures/{filename}", audio_fs, audio_pcm)
    print(f"   ✓ Audio sonification saved to: ./figures/{filename}")

# Example usage inside your run_me.py pipeline:
# sonify_heartbeats(t, ecg_normal, FS, "sonar_normal.wav")
# sonify_heartbeats(t, ecg_arrhythmia, FS, "sonar_zombie.wav")