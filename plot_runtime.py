import sys
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_and_aggregate_runtime_data(filename):
    """
    Reads a text file with lines like:
      n_bodies, n_threads, n_simulations, repetition, runtime

    Aggregates runtimes by configuration and calculates statistics for each configuration.
    Returns a list of tuples: [(n_bodies, mean_runtime, median_runtime, std_dev_runtime, min_runtime, max_runtime), ...].
    """
    runtimes = defaultdict(list)  # To store runtimes for each configuration

    # Regex to parse n_bodies, n_threads, and GPU runtime
    line_regex = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,")
    gpu_time_regex = re.compile(r"GPU parallel computation took\s+(\d+)\s+microseconds")
    #gpu_time_regex = re.compile(r"GPU total computation took\s+(\d+)\s+milliseconds")

    last_config = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # Match configuration line
            m_line = line_regex.search(line)
            if m_line:
                n_bodies = int(m_line.group(1))
                n_threads = int(m_line.group(2))
                n_simulations = int(m_line.group(3))
                repetition = int(m_line.group(4))
                last_config = (n_bodies, n_threads, n_simulations)
                continue

            # Match GPU runtime
            m_gpu_time = gpu_time_regex.search(line)
            if m_gpu_time and last_config is not None:
                runtime = int(m_gpu_time.group(1))  # runtime in microseconds
                runtimes[last_config].append(runtime)

    # Compute statistics for each configuration
    aggregated_data = []
    for (n_bodies, n_threads, n_simulations), runtime_list in runtimes.items():
        mean_runtime = np.mean(runtime_list)
        median_runtime = np.median(runtime_list)
        std_dev_runtime = np.std(runtime_list)
        min_runtime = np.min(runtime_list)
        max_runtime = np.max(runtime_list)
        aggregated_data.append((n_bodies, mean_runtime, median_runtime, std_dev_runtime, min_runtime, max_runtime))

    return aggregated_data


def plot_runtime_stats(aggregated_data, png_filename):
    """
    Plot mean and median runtime vs. number of bodies/threads with error bars (standard deviation).
    
    - aggregated_data: List of tuples [(n_bodies, mean_runtime, median_runtime, std_dev_runtime, min_runtime, max_runtime), ...].
    """
    if not aggregated_data:
        print("No runtime data to plot.")
        return

    # Sort data by n_bodies
    sorted_data = sorted(aggregated_data, key=lambda x: x[0])
    n_bodies = [d[0] for d in sorted_data]
    mean_runtimes = [d[1] / 1000 for d in sorted_data]  # Convert to milliseconds
    median_runtimes = [d[2] / 1000 for d in sorted_data]  # Convert to milliseconds
    std_devs = [d[3] / 1000 for d in sorted_data]  # Convert to milliseconds

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(n_bodies, mean_runtimes, yerr=std_devs, fmt='o', label='Mean Runtime (with std dev)')
    plt.plot(n_bodies, median_runtimes, marker='x', linestyle='--', color='red', label='Median Runtime')

    plt.xlabel("Number of Bodies/Threads")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime Statistics vs. Number of Bodies/Threads")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_filename, dpi=150)
    plt.show()
    print(f"Saved runtime statistics plot as '{png_filename}'")

###############################################################################
# Main program
###############################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <datafile.txt>")
        sys.exit(1)

    datafile = sys.argv[1]
    base = os.path.splitext(datafile)[0]

    # Parse and aggregate runtime data
    aggregated_data = parse_and_aggregate_runtime_data(datafile)
    print("Aggregated Runtime Data:")
    for entry in aggregated_data:
        n_b, mean_rt, median_rt, std_rt, min_rt, max_rt = entry
        print(f"  n_bodies = {n_b}, mean = {mean_rt:.2f} µs, median = {median_rt:.2f} µs, std dev = {std_rt:.2f} µs, min = {min_rt} µs, max = {max_rt} µs")

    # Plot runtime statistics
    plot_runtime_stats(aggregated_data, f"{base}_runtime_stats.png")
