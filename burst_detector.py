from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from more_itertools import consecutive_groups
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema


@dataclass
class BurstEvent:
    electrode: str
    start_s: float
    end_s: float
    duration_s: float
    n_spikes: int
    mean_isi_s: float
    isi_threshold_s: float


def fwhm_to_sigma(fwhm: float) -> float:
    """Convert Full Width at Half Maximum (in bins) to Gaussian sigma."""
    return fwhm / np.sqrt(8.0 * np.log(2.0))


def compute_smoothed_log_isi(
    spike_times_s: np.ndarray,
    bins: np.ndarray,
    fwhm: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bin centers and Gaussian-smoothed log10(ISI) histogram counts."""
    spikes = np.sort(np.asarray(spike_times_s, dtype=float).ravel())
    isi = np.diff(spikes)
    isi = isi[isi > 0]  # Guard against duplicate timestamps

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    if isi.size == 0:
        return bin_centers, np.zeros_like(bin_centers)

    counts, _ = np.histogram(np.log10(isi), bins=bins)
    sigma = fwhm_to_sigma(fwhm)
    smooth_counts = gaussian_filter1d(counts.astype(float), sigma=sigma, mode="nearest")
    return bin_centers, smooth_counts


def find_log_isi_threshold(
    bin_centers: np.ndarray,
    smooth_counts: np.ndarray,
    max_intraburst_isi_s: float = 0.1,
) -> tuple[float | None, np.ndarray, np.ndarray]:
    """Identify the Log-ISI valley threshold between intra- and inter-burst peaks."""
    local_minima = argrelextrema(smooth_counts, np.less)[0]
    local_maxima = argrelextrema(smooth_counts, np.greater)[0]

    if local_maxima.size == 0 or local_minima.size == 0:
        return None, local_minima, local_maxima

    # Find the primary intra-burst peak below max_intraburst_isi_s
    log_cap = np.log10(max_intraburst_isi_s)
    intra_peaks = local_maxima[bin_centers[local_maxima] <= log_cap]
    if intra_peaks.size == 0:
        return None, local_minima, local_maxima

    first_peak = intra_peaks[np.argmax(smooth_counts[intra_peaks])]
    valleys_after_peak = local_minima[local_minima > first_peak]
    if valleys_after_peak.size == 0:
        return None, local_minima, local_maxima

    threshold_s = float(10.0 ** bin_centers[valleys_after_peak[0]])
    return threshold_s, local_minima, local_maxima


def extract_bursts(
    spike_times_s: np.ndarray,
    electrode: str,
    isi_threshold_s: float,
    min_spikes: int = 3,
) -> list[BurstEvent]:
    """Group consecutive spikes with ISI <= isi_threshold_s into burst events."""
    spikes = np.sort(np.asarray(spike_times_s, dtype=float).ravel())
    if spikes.size < min_spikes:
        return []

    isi = np.diff(spikes)
    short_isi_indices = np.where(isi <= isi_threshold_s)[0]

    events: list[BurstEvent] = []
    for group in consecutive_groups(short_isi_indices):
        idx = list(group)
        start_idx = idx[0]
        end_idx = idx[-1] + 1  # Include the closing spike of the last ISI
        n_spikes = end_idx - start_idx + 1

        if n_spikes >= min_spikes:
            burst_spikes = spikes[start_idx : end_idx + 1]
            events.append(
                BurstEvent(
                    electrode=electrode,
                    start_s=float(burst_spikes[0]),
                    end_s=float(burst_spikes[-1]),
                    duration_s=float(burst_spikes[-1] - burst_spikes[0]),
                    n_spikes=int(n_spikes),
                    mean_isi_s=float(np.mean(np.diff(burst_spikes))),
                    isi_threshold_s=isi_threshold_s,
                )
            )
    return events


def save_log_isi_plot(
    bin_centers: np.ndarray,
    smooth_counts: np.ndarray,
    local_minima: np.ndarray,
    local_maxima: np.ndarray,
    threshold_s: float | None,
    electrode: str,
    out_path: Path,
) -> None:
    """Save a diagnostic Log-ISI histogram plot."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bin_centers, smooth_counts, color="#000000", lw=1.8, label="Smoothed Counts")

    if local_minima.size:
        ax.scatter(
            bin_centers[local_minima],
            smooth_counts[local_minima],
            color="#0072B2",
            zorder=3,
            label="Minima",
        )
    if local_maxima.size:
        ax.scatter(
            bin_centers[local_maxima],
            smooth_counts[local_maxima],
            color="#D55E00",
            zorder=3,
            label="Maxima",
        )
    if threshold_s is not None:
        ax.axvline(
            np.log10(threshold_s),
            color="#009E73",
            ls="--",
            lw=1.4,
            label=f"Threshold ({threshold_s * 1e3:.1f} ms)",
        )

    ax.set_xlabel(r"$\log_{10}(\mathrm{ISI\ [s]})$")
    ax.set_ylabel("Count")
    ax.set_title(f"{electrode} — Log-ISI Distribution")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_pipeline(
    hdf5_path: Path,
    output_dir: Path,
    sampling_rate: float = 25000.0,
    fwhm: float = 6.0,
    bin_size: float = 0.1,
    min_spikes: int = 3,
    save_plots: bool = True,
) -> pd.DataFrame:
    bins = np.arange(-4.0, 4.0 + bin_size, bin_size)
    all_bursts: list[BurstEvent] = []
    bad_electrodes: list[str] = []

    with h5py.File(hdf5_path, "r") as f:
        spike_group = f["spiketimes"]
        for elec_name in sorted(spike_group.keys()):
            spike_times_s = np.ravel(spike_group[elec_name][:]) / sampling_rate
            bin_centers, smooth_counts = compute_smoothed_log_isi(spike_times_s, bins, fwhm)
            threshold_s, minima, maxima = find_log_isi_threshold(bin_centers, smooth_counts)

            if save_plots:
                save_log_isi_plot(
                    bin_centers,
                    smooth_counts,
                    minima,
                    maxima,
                    threshold_s,
                    elec_name,
                    output_dir / "plots" / f"{elec_name}.png",
                )

            if threshold_s is None:
                bad_electrodes.append(elec_name)
                continue

            all_bursts.extend(extract_bursts(spike_times_s, elec_name, threshold_s, min_spikes))

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(b) for b in all_bursts])
    df.to_csv(output_dir / "burst_events.csv", index=False)
    (output_dir / "bad_electrodes.txt").write_text("\n".join(bad_electrodes))

    print(f"Detected {len(df)} bursts across {df['electrode'].nunique() if not df.empty else 0} electrodes.")
    print(f"Skipped {len(bad_electrodes)} electrodes (no valid threshold).")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect bursts from MEA spike times via Log-ISI.")
    parser.add_argument("hdf5_path", type=Path, help="Path to .mua.hdf5 file")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("./results"))
    parser.add_argument("--fs", type=float, default=25000.0, help="Sampling rate in Hz")
    parser.add_argument("--min-spikes", type=int, default=3, help="Minimum spikes per burst")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving diagnostic plots")
    args = parser.parse_args()

    run_pipeline(
        hdf5_path=args.hdf5_path,
        output_dir=args.output_dir,
        sampling_rate=args.fs,
        min_spikes=args.min_spikes,
        save_plots=not args.no_plots,
    )
