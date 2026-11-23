from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.extractors as se

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.linewidth": 1.2,
})


def plot_trace(
    file_path: Path,
    channel_index: int = 8,
    start_s: float = 8.52,
    stop_s: float = 11.52,
    stim_window_s: tuple[float, float] | None = (9.52, 10.52),
    scale_uv: float = 80.0,
    save_path: Path | None = None,
) -> None:
    """Plot a raw MEA voltage trace snippet with a stimulus bar and voltage scalebar."""
    recording = se.read_mcsraw(file_path=str(file_path))
    fs = recording.get_sampling_frequency()

    start_frame = int(round(fs * start_s))
    end_frame = int(round(fs * stop_s))
    channel_id = recording.channel_ids[channel_index]

    trace = recording.get_traces(
        start_frame=start_frame,
        end_frame=end_frame,
        channel_ids=[channel_id],
        return_scaled=True,
    ).ravel()

    # Align time axis relative to stimulus onset (or start_s if no stimulus)
    t0 = stim_window_s[0] if stim_window_s else start_s
    time_s = (np.arange(trace.size) / fs) + start_s - t0

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(time_s, trace, color="#000000", lw=0.8)

    if stim_window_s is not None:
        stim_start = stim_window_s[0] - t0
        stim_end = stim_window_s[1] - t0
        ax.axvspan(stim_start, stim_end, color="#009E73", alpha=0.25, zorder=0)

        y_top = np.percentile(trace, 99.5) + 0.15 * np.ptp(trace)
        ax.hlines(y=y_top, xmin=stim_start, xmax=stim_end, color="#000000", lw=2.5)
        ax.text(
            0.5 * (stim_start + stim_end),
            y_top + 0.04 * np.ptp(trace),
            f"Light ({stim_end - stim_start:g} s)",
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
        )

    # Vertical voltage scalebar positioned just left of the trace
    x_bar = time_s[0] - 0.03 * (time_s[-1] - time_s[0])
    y_bar_min = np.percentile(trace, 5)
    ax.vlines(x=x_bar, ymin=y_bar_min, ymax=y_bar_min + scale_uv, color="#000000", lw=2.5)
    ax.text(
        x_bar - 0.015 * (time_s[-1] - time_s[0]),
        y_bar_min + 0.5 * scale_uv,
        rf"{scale_uv:g} $\mu$V",
        ha="right",
        va="center",
        rotation=90,
        fontsize=12,
        weight="bold",
    )

    ax.set_xlabel("Time relative to stimulus (s)" if stim_window_s else "Time (s)", fontsize=13, weight="bold")
    ax.set_xlim(x_bar - 0.06 * (time_s[-1] - time_s[0]), time_s[-1])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.tick_params(axis="x", direction="out", length=6, labelsize=11)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=500, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a raw MCS MEA trace snippet.")
    parser.add_argument("file_path", type=Path, help="Path to .raw MCS file")
    parser.add_argument("--channel", type=int, default=8)
    parser.add_argument("--start", type=float, default=8.52)
    parser.add_argument("--stop", type=float, default=11.52)
    parser.add_argument("--save", type=Path, default=None, help="Output PNG path (500 DPI)")
    args = parser.parse_args()

    plot_trace(args.file_path, channel_index=args.channel, start_s=args.start, stop_s=args.stop, save_path=args.save)