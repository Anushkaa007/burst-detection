# burst-detection

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)

**Adaptive Log-ISI burst detection** and raw Multi-Electrode Array (MEA) trace visualization for extracellular electrophysiology recordings.

---

## Method

Instead of applying a fixed inter-spike interval (ISI) cutoff across heterogeneous channels, `burst-detection` derives an adaptive threshold per electrode from its firing statistics:

```mermaid
flowchart LR
    A[Sorted Spike Times] --> B["log10(ISI) Histogram"]
    B --> C[Gaussian Smoothing<br/>FWHM-parameterized]
    C --> D[Detect Intra-Burst Peak<br/>&le; 100 ms & Valley]
    D --> E[Group Consecutive Spikes<br/>ISI &le; Threshold]
    E --> F[burst_events.csv +<br/>Diagnostic Plots]
