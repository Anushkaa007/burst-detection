\# burst-detection



Python toolkit for \*\*adaptive Log-ISI burst detection\*\* from multi-electrode array (MEA) spike trains and publication-ready visualization of raw MCS voltage traces.



\## Overview



Rather than relying on a fixed inter-spike interval (ISI) cutoff, `burst\_detector.py` adapts the threshold to each electrode's firing statistics:



1\. Computes the $\\log\_{10}(\\text{ISI})$ distribution of sorted spike times.

2\. Smooths the histogram with a Gaussian kernel parameterized by Full Width at Half Maximum (FWHM).

3\. Identifies the primary intra-burst peak ($\\le 100\\text{ ms}$) and the subsequent local minimum (valley) separating intra-burst from inter-burst intervals.

4\. Groups consecutive spikes below that threshold into burst events and exports summary metrics + diagnostic plots.



\## Installation



```bash

git clone \[https://github.com/Anushkaa007/burst-detection.git](https://github.com/Anushkaa007/burst-detection.git)

cd burst-detection



\# Core burst detection

pip install -e .



\# With raw MCS trace plotting (SpikeInterface) and dev tools

pip install -e ".\[traces,dev]"

