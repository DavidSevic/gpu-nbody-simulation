#!/usr/bin/env python3
"""
This script reads a data file with lines such as:
   40000, 1024*16, 10, Loaded 40000 bodies from text files.
   ...
   GPU parallel computation took 37619500 microseconds
   GPU total computation took 1234 milliseconds.
for various thread counts.
It builds dictionaries of measured times for both the parallel and total runs,
computes averages, speedups, and efficiencies, and creates plots for both.
"""

import sys
import os
import re
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# 1) Parsing
###############################################################################

def parse_thread_count(thread_str):
    """
    Parse something like:
      '1024'    -> 1024
      '1024*16' -> 16384
      '128*2'   -> 256
    """
    thread_str = thread_str.strip()
    if '*' in thread_str:
        product = 1
        for part in thread_str.split('*'):
            product *= int(part)
        return product
    else:
        return int(thread_str)

def parse_gpu_times(filename):
    """
    Parses the given file for both GPU parallel and GPU total times.
    
    Lines we expect:
      <n_bodies>, <thread_str>, <n_simulations>, <some description text>
      ...
      GPU parallel computation took XXXX microseconds
      GPU total computation took XXXX milliseconds.
      
    This function returns two dictionaries:
      parallel_times: { thread_count : [time_in_microseconds, ...] }
      total_times:    { thread_count : [time_in_milliseconds, ...] }
    """
    # Regular expression to capture the thread line.
    # Example: "40000, 1024*16, 10, Loaded 40000 bodies ..."
    line_thread_regex = re.compile(r"^\s*(\d+)\s*,\s*([^,]+)\s*,\s*(\d+)\s*,")
    
    # Two regex patterns for the GPU times.
    parallel_regex = re.compile(r"GPU parallel computation took\s+(\d+)\s+microseconds")
    total_regex    = re.compile(r"GPU total computation took\s+(\d+)\s+milliseconds\.")
    
    parallel_times = {}
    total_times = {}
    
    last_thread = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # skip header lines if they exist
            if 'n_bodies' in line.lower():
                continue

            # Check for the thread line
            m_thread = line_thread_regex.search(line)
            if m_thread:
                thread_str = m_thread.group(2).strip()
                try:
                    last_thread = parse_thread_count(thread_str)
                except ValueError:
                    last_thread = None
                continue

            # Look for parallel timing line
            m_parallel = parallel_regex.search(line)
            if m_parallel and last_thread is not None:
                time_parallel = int(m_parallel.group(1))
                parallel_times.setdefault(last_thread, []).append(time_parallel)
                continue

            # Look for total timing line
            m_total = total_regex.search(line)
            if m_total and last_thread is not None:
                time_total = int(m_total.group(1))
                total_times.setdefault(last_thread, []).append(time_total)
                continue

    return parallel_times, total_times

###############################################################################
# 2) Compute Averages, Speedup, and Efficiency
###############################################################################

def compute_average_times(times_by_thread):
    """
    Given a dictionary { thread_count: [time1, time2, ...] },
    compute the average time for each thread_count.
    Return a list of (thread_count, avg_time).
    """
    averages = []
    for thread_count in sorted(times_by_thread.keys()):
        values = times_by_thread[thread_count]
        avg_time = sum(values) / len(values)
        averages.append((thread_count, avg_time))
    return averages

def compute_speedups(averages):
    """
    Computes speedups S(p) = T(1)/T(p) based on the averages list.
    Returns a list of (thread_count, speedup).
    """
    # Find T(1)
    t1_list = [t for (p, t) in averages if p == 1]
    if not t1_list:
        print("No T(1) found; cannot compute speedups.")
        return []
    t1 = t1_list[0]
    
    speedups = []
    for (p, t_p) in averages:
        if t_p != 0:
            s_p = t1 / t_p
            speedups.append((p, s_p))
    return speedups

def compute_efficiencies(averages):
    """
    Computes efficiencies E(p) = (T(1)/T(p)) / p.
    Returns a list of (thread_count, efficiency).
    """
    t1_list = [t for (p, t) in averages if p == 1]
    if not t1_list:
        print("No T(1) found; cannot compute efficiencies.")
        return []
    t1 = t1_list[0]
    
    efficiencies = []
    for (p, t_p) in averages:
        if t_p != 0:
            s_p = t1 / t_p
            e_p = s_p / p
            efficiencies.append((p, e_p))
    return efficiencies

###############################################################################
# 3) Plotting Functions
###############################################################################

def plot_time_numeric(averages, png_filename, time_type="GPU Time", time_unit="µs",
                      use_log_scale_x=False, use_log_scale_y=False):
    """
    Plots T(p) vs. p using the computed averages.
    The plot shows the measured average time and an 'ideal' line T(1)/p.
    
    Parameters:
      averages: List of (thread_count, avg_time)
      png_filename: Filename to save the plot image.
      time_type: A descriptor string (e.g. "Parallel" or "Total") used in labels.
      time_unit: The unit displayed on the y-axis.
    """
    if not averages:
        print(f"No data to plot for {time_type.lower()} time.")
        return

    # Ensure the data is sorted by thread_count
    data_sorted = sorted(averages, key=lambda x: x[0])
    thread_counts = [d[0] for d in data_sorted]
    avg_times = [d[1] for d in data_sorted]

    plt.figure(figsize=(10,6))
    plt.plot(thread_counts, avg_times, marker='o', label=f'Measured T(p)')

    # Plot ideal scaling line T(1)/p if available
    t1_candidates = [t for (p, t) in data_sorted if p == 1]
    if t1_candidates:
        t1 = t1_candidates[0]
        p_min = min(thread_counts)
        p_max = max(thread_counts)
        p_line = np.linspace(p_min, p_max, 100)
        p_line = p_line[p_line > 0]
        t_line = t1 / p_line
        plt.plot(p_line, t_line, 'r--', alpha=0.6, label='Ideal T(1)/p')

    # Annotate points
    for p_val, t_val in zip(thread_counts, avg_times):
        plt.annotate(f"{t_val:.2f}", xy=(p_val, t_val),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9)

    if use_log_scale_x:
        plt.xscale('log')
    if use_log_scale_y:
        plt.yscale('log')

    plt.xlabel("Threads (p)")
    plt.ylabel(f"Average {time_type} Time ({time_unit})")
    plt.title(f"{time_type} T(p) vs. Threads")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_filename, dpi=150)
    plt.show()
    print(f"Saved {time_type.lower()} time plot as '{png_filename}'")

def plot_speedup_numeric_colored_stepped(speedups, png_filename,
                                         plot_title="Speedup vs. Threads",
                                         use_log_scale_x=False):
    """
    Plots speedups (S(p)=T(1)/T(p)) vs. threads p using a stepped background.
    
    Parameters:
      speedups: List of (p, speedup)
      png_filename: Filename to save the plot image.
      plot_title: The title used in the plot.
      use_log_scale_x: If True, sets a log scale on the x-axis.
    """
    speedups_sorted = sorted(speedups, key=lambda x: x[0])
    if not speedups_sorted:
        print(f"No speedup data to plot: {png_filename}")
        return
    
    p_vals = [d[0] for d in speedups_sorted]
    s_vals = [d[1] for d in speedups_sorted]
    
    p_min = min(p_vals)
    p_max = max(p_vals)
    s_max = max(s_vals) * 1.1  # add margin
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    # Optionally, clamp the x-axis if p_max is very large
    clamp_factor = 2.0
    if p_max > clamp_factor * s_max:
        x_max_vis = max(p_max, clamp_factor * s_max)
    else:
        x_max_vis = p_max

    # Create a fine grid for the stepped background
    STEPS = 100000
    p_start = max(1, p_min)
    p_arr = np.linspace(p_start, x_max_vis, STEPS)
    
    # Fill the background in three regions:
    # Red: S(p) < 1, Green: 1 ≤ S(p) < p, Yellow: S(p) ≥ p
    ax.fill_between(p_arr, 0, 1, color='red', alpha=0.15, label='S(p) < 1', step='mid')
    y_top_green = np.minimum(p_arr, s_max)
    ax.fill_between(p_arr, 1, y_top_green, where=(p_arr >= 1),
                    color='green', alpha=0.15, label='1 ≤ S(p) < p', step='mid')
    ax.fill_between(p_arr, p_arr, s_max, where=(p_arr <= s_max),
                    color='yellow', alpha=0.15, label='S(p) ≥ p', step='mid')
    
    # Plot measured speedups
    ax.plot(p_vals, s_vals, marker='o', linestyle='-', label='Measured Speedup')
    # Plot ideal speedup line y=p as stepped dashed red line
    ax.plot(p_arr, p_arr, 'r--', alpha=0.8, label='Ideal: y = p', drawstyle='steps-mid')
    
    # Annotate each point
    for x_val, y_val in zip(p_vals, s_vals):
        ax.annotate(f"{y_val:.2f}", xy=(x_val, y_val), xytext=(0, 5),
                    textcoords='offset points', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlim(left=p_start, right=x_max_vis)
    ax.set_ylim(bottom=0, top=max(s_max, 1))
    ax.set_xlabel("Threads (p)")
    ax.set_ylabel("Speedup S(p) = T(1)/T(p)")
    ax.set_title(plot_title)
    if use_log_scale_x:
        ax.set_xscale('log')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(png_filename, dpi=150)
    plt.show()
    print(f"Saved speedup plot as '{png_filename}'")

def plot_efficiency_numeric(efficiencies, png_filename,
                            plot_title="Efficiency vs. Threads",
                            use_log_scale_x=False):
    """
    Plots efficiency E(p)=S(p)/p vs. threads p.
    
    Parameters:
      efficiencies: List of (p, efficiency)
      png_filename: Filename to save the plot image.
      plot_title: The title used in the plot.
      use_log_scale_x: If True, uses logarithmic scaling on the x-axis.
    """
    if not efficiencies:
        print(f"No efficiency data to plot: {png_filename}")
        return

    sorted_data = sorted(efficiencies, key=lambda x: x[0])
    thread_counts = [d[0] for d in sorted_data]
    e_values = [d[1] for d in sorted_data]

    plt.figure(figsize=(10,6))
    plt.plot(thread_counts, e_values, marker='o', linestyle='-')

    for p, e in zip(thread_counts, e_values):
        plt.annotate(f"{e:.2f}", xy=(p, e), xytext=(0,5),
                     textcoords='offset points', ha='center', va='bottom', fontsize=9)

    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Ideal E=1.0')
    plt.xlabel("Threads (p)")
    plt.ylabel("Efficiency E(p)")
    plt.title(plot_title)
    plt.grid(True)
    plt.legend()
    if use_log_scale_x:
        plt.xscale('log')
    plt.tight_layout()
    plt.savefig(png_filename, dpi=150)
    plt.show()
    print(f"Saved efficiency plot as '{png_filename}'")

###############################################################################
# 4) Main Program
###############################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_gpu_times.py <datafile.txt>")
        sys.exit(1)

    datafile = sys.argv[1]
    base = os.path.splitext(datafile)[0]

    # Parse data for both parallel and total timings.
    parallel_times, total_times = parse_gpu_times(datafile)
    
    # Compute averages for parallel timings (microseconds)
    parallel_averages = compute_average_times(parallel_times)
    print("Parallel Averages:")
    for thr, t in parallel_averages:
        print(f"  Threads = {thr}, T_parallel = {t:.2f} µs")

    # Compute averages for total timings (milliseconds)
    total_averages = compute_average_times(total_times)
    print("\nTotal Averages:")
    for thr, t in total_averages:
        print(f"  Threads = {thr}, T_total = {t:.2f} ms")

    # Compute speedups and efficiencies for parallel times
    parallel_speedups = compute_speedups(parallel_averages)
    parallel_efficiencies = compute_efficiencies(parallel_averages)

    # Compute speedups and efficiencies for total times
    total_speedups = compute_speedups(total_averages)
    total_efficiencies = compute_efficiencies(total_averages)
    """
    # Plot GPU Parallel Runtime vs. Threads
    plot_time_numeric(
        parallel_averages,
        f"{base}_parallel_time.png",
        time_type="GPU Parallel",
        time_unit="µs",
        use_log_scale_x=True,
        use_log_scale_y=False
    )
    """
    """
    # Plot GPU Total Runtime vs. Threads
    plot_time_numeric(
        total_averages,
        f"{base}_total_time.png",
        time_type="GPU Total",
        time_unit="ms",
        use_log_scale_x=True,
        use_log_scale_y=False
    )
    """
    # Plot speedup for parallel timing
    plot_speedup_numeric_colored_stepped(
        parallel_speedups,
        f"{base}_parallel_speedup.png",
        plot_title="Speedup (GPU Parallel) vs. Threads",
        use_log_scale_x=True
    )
    
    # Plot efficiency for parallel timing
    plot_efficiency_numeric(
        parallel_efficiencies,
        f"{base}_parallel_efficiency.png",
        plot_title="Efficiency (GPU Parallel) vs. Threads",
        use_log_scale_x=True
    )
    
    # (Optional) Also plot speedup and efficiency for total times.
    # Depending on your application these may be less meaningful.
    plot_speedup_numeric_colored_stepped(
        total_speedups,
        f"{base}_total_speedup.png",
        plot_title="Speedup (GPU Total) vs. Threads",
        use_log_scale_x=True
    )
    
    plot_efficiency_numeric(
        total_efficiencies,
        f"{base}_total_efficiency.png",
        plot_title="Efficiency (GPU Total) vs. Threads",
        use_log_scale_x=True
    )
