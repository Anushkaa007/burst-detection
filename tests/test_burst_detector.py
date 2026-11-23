from __future__ import annotations

import numpy as np
import pytest

from burst_detection.burst_detector import (
    compute_smoothed_log_isi,
    extract_bursts,
    find_log_isi_threshold,
    fwhm_to_sigma,
)

BINS = np.arange(-4.0, 4.1, 0.1)


def make_synthetic_burst_train(
    n_bursts: int = 30,
    spikes_per_burst: int = 6,
    intra_isi_s: float = 0.010,
    inter_burst_s: float = 2.0,
    jitter_s: float = 0.001,
    seed: int = 42,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Generate a bimodal spike train with known ground-truth burst windows."""
    rng = np.random.default_rng(seed)
    spikes: list[float] = []
    ground_truth: list[tuple[float, float]] = []

    t = 1.0
    for _ in range(n_bursts):
        burst_start = t
        burst_spikes = [t]
        for _ in range(spikes_per_burst - 1):
            t += intra_isi_s + rng.uniform(-jitter_s, jitter_s)
            burst_spikes.append(t)
        spikes.extend(burst_spikes)
        ground_truth.append((burst_start, burst_spikes[-1]))
        t += inter_burst_s + rng.uniform(-0.1, 0.1)

    return np.array(spikes, dtype=float), ground_truth


def test_fwhm_to_sigma():
    sigma = fwhm_to_sigma(2.354820045)
    assert np.isclose(sigma, 1.0, atol=1e-6)


def test_synthetic_burst_recovery():
    spikes, ground_truth = make_synthetic_burst_train(
        n_bursts=30,
        spikes_per_burst=6,
        intra_isi_s=0.010,
        inter_burst_s=2.0,
    )

    bin_centers, smooth_counts = compute_smoothed_log_isi(spikes, BINS, fwhm=6.0)
    threshold_s, _, _ = find_log_isi_threshold(bin_centers, smooth_counts)

    assert threshold_s is not None
    assert 0.02 < threshold_s < 1.0  # Threshold must fall cleanly between 10 ms and 2 s

    events = extract_bursts(spikes, electrode="elec_0", isi_threshold_s=threshold_s, min_spikes=3)
    assert len(events) == len(ground_truth)

    for event, (gt_start, gt_end) in zip(events, ground_truth):
        assert event.n_spikes == 6
        assert np.isclose(event.start_s, gt_start, atol=1e-9)
        assert np.isclose(event.end_s, gt_end, atol=1e-9)
        assert np.isclose(event.duration_s, gt_end - gt_start, atol=1e-9)


def test_unsorted_and_duplicate_spikes_handled_safely():
    spikes, _ = make_synthetic_burst_train(n_bursts=10, spikes_per_burst=5)
    # Inject duplicates and shuffle order
    messy_spikes = np.concatenate([spikes, spikes[:5]])
    rng = np.random.default_rng(7)
    rng.shuffle(messy_spikes)

    _, smooth_counts = compute_smoothed_log_isi(messy_spikes, BINS, fwhm=6.0)
    assert np.all(np.isfinite(smooth_counts))
    assert smooth_counts.sum() > 0


@pytest.mark.parametrize("spikes", [np.array([]), np.array([1.0]), np.array([1.0, 1.005])])
def test_empty_and_short_spike_trains(spikes: np.ndarray):
    bin_centers, smooth_counts = compute_smoothed_log_isi(spikes, BINS, fwhm=6.0)
    threshold_s, _, _ = find_log_isi_threshold(bin_centers, smooth_counts)
    events = extract_bursts(spikes, electrode="elec_0", isi_threshold_s=0.05, min_spikes=3)

    assert threshold_s is None
    assert events == []


def test_extract_bursts_respects_min_spikes():
    # One 2-spike doublet (should be ignored) and one 4-spike burst (should be kept)
    spikes = np.array([1.00, 1.01, 5.00, 5.01, 5.02, 5.03])
    events = extract_bursts(spikes, electrode="elec_1", isi_threshold_s=0.05, min_spikes=3)

    assert len(events) == 1
    assert events[0].n_spikes == 4
    assert np.isclose(events[0].start_s, 5.00)
    assert np.isclose(events[0].end_s, 5.03)
