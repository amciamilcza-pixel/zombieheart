import os
import numpy as np
import matplotlib.pyplot as plt

from real_ecg_loader import load_real_ecg, invert_ecg
from dft_filter import recover_ecg, heart_rate_from_dft


def classify_heart(bpm, is_inverted=False):
    if is_inverted and bpm < 60:
        return "Zombie / dying heart"
    if bpm < 60:
        return "Slow heart / bradycardia"
    if bpm > 100:
        return "Fast heart / tachycardia"
    return "Normal heart"


def analyse_one_ecg(title, record_name, start_sec, duration_sec, invert=False):
    print("\n" + "=" * 60)
    print(title)

    # Load ECG
    t, ecg, fs = load_real_ecg(
        record_name=record_name,
        start_sec=start_sec,
        duration_sec=duration_sec,
        channel=0
    )

    # Optional inversion
    if invert:
        ecg = invert_ecg(ecg)

    # Filter / recover
    filtered_ecg, freqs, mag_before, mag_after = recover_ecg(ecg, fs)

    # BPM estimation
    bpm = heart_rate_from_dft(filtered_ecg, fs)
    label = classify_heart(bpm, is_inverted=invert)

    print(f"Record used: {record_name}")
    print(f"Start time: {start_sec} s")
    print(f"Duration: {duration_sec} s")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Estimated BPM: {bpm:.1f}")
    print(f"Classification: {label}")

    return {
        "title": title,
        "record_name": record_name,
        "t": t,
        "raw": ecg,
        "filtered": filtered_ecg,
        "freqs": freqs,
        "mag_before": mag_before,
        "mag_after": mag_after,
        "bpm": bpm,
        "label": label
    }


def style_axis(ax):
    ax.set_facecolor("#0b0b0b")
    ax.tick_params(colors="lightgray", labelsize=9)

    for spine in ax.spines.values():
        spine.set_color("#444444")

    ax.grid(True, color="gray", alpha=0.18, linestyle="--")


def get_theme(title):
    """
    Choose colors based on the ECG type.
    """

    title_lower = title.lower()

    if "normal" in title_lower:
        return {
            "main": "#00ff88",       # bright green
            "soft": "#7dffbd",       # softer green
            "title": "#00ff88",
            "name": "NORMAL"
        }

    if "fast" in title_lower:
        return {
            "main": "#ff3333",       # red
            "soft": "#ff9a9a",       # softer red
            "title": "#ff3333",
            "name": "FAST / DYING"
        }

    if "zombie" in title_lower or "reversed" in title_lower:
        return {
            "main": "#b000ff",       # purple
            "soft": "#d88cff",       # softer purple
            "title": "#b000ff",
            "name": "ZOMBIE REVERSED"
        }

    return {
        "main": "#ffd400",
        "soft": "#eeeeee",
        "title": "#ffd400",
        "name": "ECG"
    }


def plot_summary(results):
    os.makedirs("figures", exist_ok=True)

    plt.style.use("dark_background")

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16, 13),
        constrained_layout=True
    )

    fig.patch.set_facecolor("#050505")

    # Extra spacing control
    fig.set_constrained_layout_pads(
        w_pad=0.25,
        h_pad=0.25,
        hspace=0.14,
        wspace=0.08
    )

    raw_color = "#777777"

    for i, result in enumerate(results):
        theme = get_theme(result["title"])

        t = result["t"]
        raw = result["raw"]
        filtered = result["filtered"]
        freqs = result["freqs"]
        mag_before = result["mag_before"]
        mag_after = result["mag_after"]
        bpm = result["bpm"]
        label = result["label"]
        title = result["title"]

        # Centre both signals so raw and filtered are directly comparable
        raw_plot = raw - np.mean(raw)
        filtered_plot = filtered - np.mean(filtered)

        # Normalize spectra slightly for cleaner visual comparison
        # This keeps the shape but avoids one spectrum visually dominating too much.
        mag_before_plot = mag_before
        mag_after_plot = mag_after

        # =========================
        # Left column: time signal
        # =========================
        ax1 = axes[i, 0]
        style_axis(ax1)

        ax1.plot(
            t,
            raw_plot,
            color=raw_color,
            linewidth=1.0,
            alpha=0.75,
            label="Raw ECG"
        )

        ax1.plot(
            t,
            filtered_plot,
            color=theme["main"],
            linewidth=1.7,
            label="Filtered ECG"
        )

        ax1.set_title(
            f"{theme['name']} ECG  |  {label}  |  BPM = {bpm:.1f}",
            color=theme["title"],
            fontsize=11,
            pad=8
        )

        ax1.set_xlabel("Time [s]", color="lightgray", fontsize=10, labelpad=4)
        ax1.set_ylabel("Amplitude [mV]", color="lightgray", fontsize=10, labelpad=4)

        ax1.legend(
            facecolor="#111111",
            edgecolor="#333333",
            fontsize=9,
            loc="upper right"
        )

        # =========================
        # Right column: spectrum
        # =========================
        ax2 = axes[i, 1]
        style_axis(ax2)

        ax2.plot(
            freqs,
            mag_before_plot,
            color=raw_color,
            linewidth=1.0,
            alpha=0.7,
            label="Before filtering"
        )

        ax2.plot(
            freqs,
            mag_after_plot,
            color=theme["main"],
            linewidth=1.4,
            label="After filtering"
        )

        ax2.set_xlim(0, 60)

        ax2.set_title(
            f"Spectrum: {theme['name']} ECG",
            color=theme["title"],
            fontsize=11,
            pad=8
        )

        ax2.set_xlabel("Frequency [Hz]", color="lightgray", fontsize=10, labelpad=4)
        ax2.set_ylabel("|X(f)|", color="lightgray", fontsize=10, labelpad=4)

        ax2.legend(
            facecolor="#111111",
            edgecolor="#333333",
            fontsize=9,
            loc="upper right"
        )

    fig.suptitle(
        "Real ECG Analysis Summary",
        color="#ff9f00",
        fontsize=18,
        y=1.02
    )

    save_path = "figures/real_ecg_summary.png"

    plt.savefig(
        save_path,
        dpi=300,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight"
    )

    print(f"\nSaved figure to: {save_path}")

    plt.show()

def main():
    results = []

    # 1. Normal ECG
    results.append(analyse_one_ecg(
        title="Normal real ECG",
        record_name="100",
        start_sec=0,
        duration_sec=10,
        invert=False
    ))

    # 2. Fast ECG
    results.append(analyse_one_ecg(
        title="Fast real ECG",
        record_name="209",
        start_sec=861,
        duration_sec=10,
        invert=False
    ))

    # 3. Slow ECG reversed for zombie effect
    results.append(analyse_one_ecg(
        title="Zombie reversed ECG",
        record_name="232",
        start_sec=0,
        duration_sec=10,
        invert=True
    ))

    # Make the combined plot
    plot_summary(results)


if __name__ == "__main__":
    main()