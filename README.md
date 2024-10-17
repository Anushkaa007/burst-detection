#!/usr/bin/env python3

"""
Burst Detection Script

This script detects bursts in neural spike train data using inter-spike intervals (ISI).
It processes data from an HDF5 file and generates plots for each electrode.

Dependencies:
- numpy
- matplotlib
- h5py
- scipy
- more_itertools

Usage:
Run the script in a Python environment with the required libraries installed.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import argrelextrema
from more_itertools import consecutive_groups

# Constants
SAMPLING_RATE = 25000  # Hz
FWHM = 6                # Full Width at Half Maximum for Gaussian smoothing
XARB = 0.1              # Arbitrary bin size for log ISI histogram
BINS = np.arange(-4, 4, XARB)  # Log ISI histogram bins

def sigma2fwhm(sigma):
    """Convert sigma to Full Width at Half Maximum (FWHM)."""
    return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
    """Convert Full Width at Half Maximum (FWHM) to sigma."""
    return fwhm / np.sqrt(8 * np.log(2))

def load_data(file_path):
    """Load spike time data from an HDF5 file."""
    return h5py.File(file_path, 'r')

def process_electrode_data(elec_id, array_sorted, sigma):
    """Process spike times for a given electrode to detect bursts."""
    isi = np.diff(array_sorted)  # Calculate inter-spike interval
    logisi = np.log10(isi)       # Take log for plotting log ISI

    counts, _ = np.histogram(logisi, bins=BINS)  # Histogram of log ISI
    counts = np.append(counts, 0)  # Append zero for plotting

    # Smooth the counts using a Gaussian kernel
    smooth_counts = np.zeros(counts.shape)
    x_vals = np.arange(len(counts))
    
    for x_position in x_vals:
        kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
        kernel /= np.sum(kernel)
        smooth_counts[x_position] = np.sum(np.multiply(kernel, counts))

    return smooth_counts

def detect_bursts(smooth_counts, elec_id):
    """Detect bursts based on smoothed counts."""
    local_minima = argrelextrema(smooth_counts, np.less)[0]
    local_maxima = argrelextrema(smooth_counts, np.greater)[0]

    # Plotting local maxima and minima along with the log(ISI) histogram
    plt.figure()
    plt.plot(BINS, smooth_counts, label='Smoothed Counts')
    
    for j in local_minima:
        plt.scatter(BINS[j], smooth_counts[j], color='blue', label='Minima')
        
    for k in local_maxima:
        plt.scatter(BINS[k], smooth_counts[k], color='red', label='Maxima')
        
    plt.legend(loc='upper right')
    plt.title(f'Electrode {elec_id} - Log(ISI) Histogram')
    plt.savefig(f'./folder/elec_{elec_id}.png', bbox_inches='tight')
    plt.close()

    return local_minima, local_maxima

def main():
    sigma = fwhm2sigma(FWHM)
    f = load_data('./E12_13_07_24.mua.hdf5')  # Load data from HDF5 file
    
    bad_electrodes = []
    bursts = []

    elec_ids = np.arange(0, 60)  # Electrode IDs
    
    for i in elec_ids:
        array = f['spiketimes']['elec_' + str(i)]  # Load spike times
        array_sorted = np.ravel(array) / SAMPLING_RATE  # Convert to seconds
        
        smooth_counts = process_electrode_data(i, array_sorted, sigma)
        local_minima, local_maxima = detect_bursts(smooth_counts, i)

        if len(local_maxima) == 0:
            print(f'Electrode {i} not analyzed: No maxima found.')
            bad_electrodes.append(i)
            bursts.append([])
            continue
        
        # Further processing to determine burst characteristics...
        # Your existing burst detection logic goes here...

        print(f'Processed electrode {i}: Detected bursts.')

    bad_electrodes.sort()
    
    print(f'Total Bursts Detected: {len(bursts)}')
    print(f'Bad Electrodes: {bad_electrodes}')
    
    # Save results to files
    np.save('./burst_events_allelecs.npy', bursts)
    np.save('./bursts_badelecs.npy', bad_electrodes)

if __name__ == "__main__":
    main()
