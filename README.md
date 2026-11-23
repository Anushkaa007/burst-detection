# burst-detection

[![CI](https://github.com/Anushkaa007/burst-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Anushkaa007/burst-detection/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.exer.2025.110593-blue.svg)](https://doi.org/10.1016/j.exer.2025.110593)

**Adaptive Log-ISI burst detection** and raw Multi-Electrode Array (MEA) trace visualization for extracellular electrophysiology recordings.

---

## Method

Instead of applying a fixed inter-spike interval (ISI) cutoff across heterogeneous channels, `burst-detection` derives an adaptive threshold per electrode from its firing statistics:

```mermaid
flowchart LR
    A[Sorted Spike Times] --> B["log10(ISI) Histogram"]
    B --> C["Gaussian Smoothing<br/>(FWHM-parameterized)"]
    C --> D["Detect Intra-Burst Peak<br/>(<= 100 ms) and Valley"]
    D --> E["Group Consecutive Spikes<br/>(ISI <= Threshold)"]
    E --> F["burst_events.csv +<br/>Diagnostic Plots"]
```

## Repository Structure

```text
burst-detection/
├── .github/workflows/ci.yml    # Tests + lint on every push / PR
├── src/burst_detection/
│   ├── burst_detector.py       # Adaptive Log-ISI burst detection pipeline & CLI
│   └── plot_trace.py           # Publication-ready raw MCS voltage trace plotter
├── tests/
│   └── test_burst_detector.py  # Unit tests with synthetic ground-truth spike trains
├── CITATION.cff                # GitHub citation metadata
├── pyproject.toml              # Package metadata and dependencies
└── LICENSE                     # MIT License
```

## Installation

```bash
git clone https://github.com/Anushkaa007/burst-detection.git
cd burst-detection

# Core burst detection package
pip install -e .

# Optional: raw MCS trace plotting (SpikeInterface) and test suite
pip install -e ".[traces,dev]"
```

## Usage

### 1. Command-Line Interface (HDF5 Spike Times)

Expects an HDF5 file with datasets stored under `/spiketimes/elec_<id>` as 1D arrays of spike sample indices.

```bash
python -m burst_detection.burst_detector path/to/recording.mua.hdf5 \
    -o ./results \
    --fs 25000 \
    --min-spikes 3
```

### 2. Python API

```python
import numpy as np
from burst_detection.burst_detector import (
    compute_smoothed_log_isi,
    extract_bursts,
    find_log_isi_threshold,
)

# spike_times_s: sorted 1D array of spike times in seconds for one electrode
bins = np.arange(-4.0, 4.1, 0.1)
bin_centers, smooth_counts = compute_smoothed_log_isi(spike_times_s, bins, fwhm=6.0)
threshold_s, _, _ = find_log_isi_threshold(bin_centers, smooth_counts)

if threshold_s is not None:
    events = extract_bursts(spike_times_s, electrode="elec_0", isi_threshold_s=threshold_s)
```

### 3. Raw MCS Trace Plotting

```bash
python -m burst_detection.plot_trace path/to/recording.raw \
    --channel 8 \
    --start 8.52 \
    --stop 11.52 \
    --save ./results/ch8_trace.png
```

## Output Schema (`burst_events.csv`)

| Column | Type | Description |
| --- | --- | --- |
| `electrode` | `str` | Electrode dataset key (e.g., `elec_0`) |
| `start_s` | `float` | First spike timestamp in the burst (s) |
| `end_s` | `float` | Last spike timestamp in the burst (s) |
| `duration_s` | `float` | Total burst duration (`end_s - start_s`, s) |
| `n_spikes` | `int` | Number of spikes in the burst |
| `mean_isi_s` | `float` | Mean intra-burst ISI (s) |
| `isi_threshold_s` | `float` | Adaptive Log-ISI threshold for this electrode (s) |

Electrodes lacking a detectable intra-burst peak/valley are logged in `bad_electrodes.txt`.

## Verification

Run the synthetic ground-truth unit test suite:

```bash
pytest
```

## Citation

This repository contains the Log-ISI burst detection and MEA trace visualization code developed during my summer research project, which contributed to the following publication:

> Krishnan, A., Deepak, C. S., **Sarkar, A.**, & Narayan, K. S. (2025). *Spatiotemporal analysis of chick retinal waves: GABAergic regulation of wave dynamics.* **Experimental Eye Research**, 260, 110593. https://doi.org/10.1016/j.exer.2025.110593 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/40876617/))

If you use this code in your research, please cite:

```bibtex
@article{krishnan2025spatiotemporal,
  author  = {Krishnan, Abhijith and Deepak, C. S. and Sarkar, Anushka and Narayan, K. S.},
  title   = {Spatiotemporal analysis of chick retinal waves: GABAergic regulation of wave dynamics},
  journal = {Experimental Eye Research},
  volume  = {260},
  pages   = {110593},
  year    = {2025},
  doi     = {10.1016/j.exer.2025.110593},
  pmid    = {40876617}
}
```

## License

Released under the [MIT License](LICENSE).