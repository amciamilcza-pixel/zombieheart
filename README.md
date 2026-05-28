# zombieheart

ZombieHeart is a Signals and Systems project that demonstrates how the Discrete Fourier Transform (DFT) can be used to analyse and recover ECG signals in a noisy “zombie apocalypse” scenario.

The project generates three synthetic ECG signals:

* **Normal heart**: a regular 70 BPM heartbeat.
* **Bradycardia**: a slow, weak, inverted heartbeat representing a “dying” heart.
* **Arrhythmia**: a fast, irregular “zombie” heartbeat with random timing and amplitude changes.

After generating the clean ECG signals, the code adds realistic noise, including random white noise, slow baseline drift, and 50 Hz power-line interference. The DFT is then used to move the signal into the frequency domain, remove unwanted frequency components, and reconstruct the cleaned ECG signal using the inverse DFT.

## Main idea

In the time domain, a noisy ECG can look messy and almost unreadable.
In the frequency domain, the useful heartbeat frequencies and the unwanted noise frequencies can be separated more clearly.

This project uses that idea to:

1. Generate ECG-like heartbeat signals.
2. Add strong noise to simulate a difficult measurement environment.
3. Use DFT-based filtering to recover the heartbeat.
4. Estimate heart rate in BPM.
5. Visualise the results with plots.
6. Convert the ECG signals into audio files.

## Files

* `run_me.py`
  Main script. Run this file to generate all figures, audio files, and printed results.

* `ecg_signals.py`
  Generates the normal, bradycardia, and arrhythmia ECG signals. Also adds noise to the signals.

* `dft_filter.py`
  Contains the DFT-based filtering functions, including bandpass filtering, notch filtering, spectrum calculation, and heart-rate estimation.

* `plots.py`
  Creates and saves all project figures in the `figures/` folder.

* `stress_tests.py`
  Contains tests for window-size sensitivity, noise robustness, and DFT limitations.

* `idk.py`
  Experimental ECG-to-audio sonification function.

* `initialize.py`
  Experimental file for loading real ECG records using WFDB.

* `heartanaylize.py`
  Placeholder file, not currently used.

## How to run

Install the required Python packages:

```bash
pip install numpy scipy matplotlib
```

Then run:

```bash
python run_me.py
```

The generated plots and audio files will be saved in the `figures/` folder.

## Output

The project creates figures showing:

* The three ECG types in the time domain and frequency domain.
* The DFT recovery process before and after filtering.
* How the DFT result changes with different window sizes.
* How noise level affects heart-rate estimation.
* Cases where the DFT becomes less reliable, such as irregular arrhythmia signals.

It also generates `.wav` audio files so the ECG signals can be heard as sound.

## Notes

Some experimental files may need cleanup before final submission. In particular, `stress_tests.py` should be checked for function-name consistency with the rest of the project.
