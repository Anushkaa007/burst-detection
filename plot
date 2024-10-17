import numpy as np
import matplotlib.pyplot as plt
import spikeextractors as se
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.dimension import _Dimension, _PREFIXES_FACTORS, _LATEX_MU

# Configure Matplotlib parameters for better aesthetics
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

class TimeDimension(_Dimension):
    """Custom time dimension class for units in seconds."""
    def __init__(self):
        super().__init__("s")
        for prefix, factor in _PREFIXES_FACTORS.items():
            latexrepr = _LATEX_MU + "s" if prefix in ["\u00b5", "u"] else None
            self.add_units(prefix + "s", factor, latexrepr)

def load_recording(file_path):
    """Load the raw recording data."""
    return se.MCSRawRecordingExtractor(filename=file_path)

def plot_trace(recording, start_time, stop_time):
    """Plot the trace snippet from the recording."""
    fs = recording.get_sampling_frequency()
    
    # Generate time array and extract trace snippet
    time_array = np.linspace(start_time, stop_time, int(fs * (stop_time - start_time)))
    trace_snippet = recording.get_traces(start_frame=int(fs * start_time), end_frame=int(fs * stop_time))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot a specific channel's trace (e.g., channel 8)
    ax.plot(time_array, trace_snippet[8], color='black')
    
    # Highlight a specific time region
    ax.axvspan(9.52, 10.52, ymax=0.85, color='green', alpha=0.5)

    # Set axis properties
    ax.tick_params(axis='x', which='major', direction='out', length=10)
    ax.set_xlabel('Time (s)', weight='bold', size=32)
    ax.set_xticks([8.52, 9.52, 10.52, 11.52])
    ax.set_xlim([8.32, 11.52])
    
    # Customize x-tick labels
    labels = ['-1', '0', '1', '2']
    ax.set_xticklabels(labels)
    
    # Draw vertical and horizontal lines with annotations
    ax.vlines(x=8.4, ymin=-60, ymax=20, color='black', linestyle='solid', linewidth=4)
    plt.annotate(r'80 $\mu$V', xy=(8.4, -25), rotation=-90, fontsize=54)
    
    ax.hlines(y=40, xmin=9.52, xmax=10.52, color='black', linestyle='solid', linewidth=4)
    plt.annotate('Light (1s)', xy=(9.65, 30), fontsize=54)

    # Hide y-axis
    ax.axes.get_yaxis().set_visible(False)

    plt.show()

def main():
    """Main function to execute the analysis."""
    file_path = './P0_10_03_22_1/P0_10_03_22_1.raw'
    
    # Load the recording data
    recording = load_recording(file_path)
    
    # Define time intervals for plotting
    start_time = 8.52  # seconds
    stop_time = 11.52  # seconds
    
    # Plot the trace from the recording
    plot_trace(recording, start_time, stop_time)

if __name__ == "__main__":
    main()
