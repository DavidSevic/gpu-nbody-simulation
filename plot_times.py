#!/usr/bin/env python3

import sys
import os
import re
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# 1) Parsing: handle lines like "40000, 1024*16, 10, Loaded 40000 bodies ..."
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

def parse_gpu_important_times(filename):
    """
    Reads a text file with lines such as:
      40000, 1, 10, Loaded 40000 bodies from text files.
      (then some blank lines, CPU lines, GPU lines, etc)
      GPU important computation took XXXX microseconds

    We'll store the times by thread count in a dict:
      times_by_thread = {
         1: [ 37619500, 38136780, ... ],
         2: [ ... ],
         ...
      }
    """
    times_by_thread = {}

    # Regex capturing 3 comma-separated fields, so the 2nd can include '*'
    # Example: "40000, 1024*16, 10, ..."
    line_thread_regex = re.compile(r"^\s*(\d+)\s*,\s*([^,]+)\s*,\s*(\d+)\s*,")
    gpu_important_regex = re.compile(r"GPU important computation took\s+(\d+)\s+microseconds")
    #gpu_important_regex = re.compile(r"GPU total computation took\s+(\d+)\s+milliseconds.")

    last_thread = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines or the header line
            if not line:
                continue
            if 'n_bodies' in line.lower():
                # Skip the header: "n_bodies, n_threads, n_simulations, runtime"
                continue

            # Check if this line indicates threads
            m_thread = line_thread_regex.search(line)
            if m_thread:
                thread_str = m_thread.group(2).strip()
                try:
                    last_thread = parse_thread_count(thread_str)
                except ValueError:
                    last_thread = None
                continue

            # Otherwise, look for "GPU important computation took XXXX microseconds"
            m_gpu = gpu_important_regex.search(line)
            if m_gpu and last_thread is not None:
                gpu_time_us = int(m_gpu.group(1))
                times_by_thread.setdefault(last_thread, []).append(gpu_time_us)

    return times_by_thread

###############################################################################
# 2) Averages, Speedup, and Efficiency
###############################################################################

def compute_average_times(times_by_thread):
    """
    Given {thread_count: [time1, time2, ...]}, compute the average time for each
    thread_count. Return a list of (thread_count, avg_time).
    """
    averages = []
    for thread_count in sorted(times_by_thread.keys()):
        values = times_by_thread[thread_count]
        avg_time = sum(values) / len(values)
        averages.append((thread_count, avg_time))
    return averages

def compute_speedups(averages):
    """
    If T(1) is the time for 1-thread, then S(P) = T(1)/T(P).
    Return a list of (P, S(P)).
    """
    # Find T(1)
    t1_list = [v for (p, v) in averages if p == 1]
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
    Parallel Efficiency E(P) = S(P)/p = [T(1)/T(p)] / p.
    Returns a list of (P, E(P)).
    """
    # Find T(1)
    t1_list = [v for (p, v) in averages if p == 1]
    if not t1_list:
        print("No T(1) found; cannot compute efficiencies.")
        return []

    t1 = t1_list[0]

    efficiencies = []
    for (p, t_p) in averages:
        if t_p != 0:
            s_p = t1 / t_p  # Speedup
            e_p = s_p / p   # Efficiency
            efficiencies.append((p, e_p))
    return efficiencies

###############################################################################
# 3) Plotting
###############################################################################

def plot_time_numeric(averages, png_filename,
                      use_log_scale_x=False,
                      use_log_scale_y=False):
    """
    Plot T(P) vs P on a numeric x-axis.
    
    - averages: list of (thread_count, avg_time)
    - use_log_scale_x: if True, the x-axis is in log scale
    - use_log_scale_y: if True, the y-axis is in log scale
    - Plots an 'ideal' line T_ideal(p) = T(1)/p as dashed red,
      so you can compare with perfect linear scaling (S(p)=p).
    """
    if not averages:
        print("No data to plot for time.")
        return

    # Sort data by thread_count ascending
    data_sorted = sorted(averages, key=lambda x: x[0])
    thread_counts = [d[0] for d in data_sorted]
    avg_times = [d[1] for d in data_sorted]

    # Create the figure
    plt.figure(figsize=(10,6))
    plt.plot(thread_counts, avg_times, marker='o', label='Measured T(p)')

    # If T(1) is available, we can plot the ideal line T_ideal(p) = T(1)/p
    t1_candidates = [t for (p, t) in data_sorted if p == 1]
    if t1_candidates:
        t1 = t1_candidates[0]
        p_min = min(thread_counts)
        p_max = max(thread_counts)

        p_line = np.linspace(p_min, p_max, 100)
        # Avoid dividing by 0 if p_min=0
        p_line = p_line[p_line > 0]
        t_line = t1 / p_line  # ideal time under perfect scaling
        plt.plot(p_line, t_line, 'r--', alpha=0.6, label='Ideal T(1)/p')

    # Annotate each data point with its T(p) value
    for p_val, t_val in zip(thread_counts, avg_times):
        plt.annotate(
            f"{t_val:.2f}",
            xy=(p_val, t_val),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=9
        )

    # Optional: use log scale for x-axis if requested
    if use_log_scale_x:
        plt.xscale('log')
    # Optional: use log scale for y-axis if requested
    if use_log_scale_y:
        plt.yscale('log')

    plt.xlabel("Threads (P) - Numeric scale")
    plt.ylabel("Average GPU Important Time (microseconds)")
    title = "T(P) vs. P"
    if use_log_scale_x:
        title += " [Log X]"
    if use_log_scale_y:
        title += " [Log Y]"
    plt.title(title)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_filename, dpi=150)
    plt.show()

    print(f"Saved numeric time plot as '{png_filename}'")


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_speedup_numeric_colored_stepped(speedups, png_filename, use_log_scale_x=False):
    """
    Plots speedups (S(p) = T(1)/T(p)) vs number of threads p in a single figure,
    using three 'stepped' background regions but with finer steps.

    Regions:
      1) Red    : 0 <= y < 1
      2) Green  : 1 <= y < x
      3) Yellow : x <= y

    The boundary between green and yellow follows the dashed red line y = x,
    drawn with a 'stepped' style. The steps are smaller than integer increments
    to make it visually smoother.

    :param speedups:        List of tuples (p, S(p)), e.g. [(1,1.0),(2,2.01),(4,3.96),...]
    :param png_filename:    Output image filename
    :param use_log_scale_x: If True, use a log scale on the x-axis
    """

    # 1) Sort data by ascending p
    speedups_sorted = sorted(speedups, key=lambda x: x[0])
    if not speedups_sorted:
        print(f"No speedup data to plot: {png_filename}")
        return

    # Extract p, s arrays
    p_vals = [t[0] for t in speedups_sorted]
    s_vals = [t[1] for t in speedups_sorted]

    p_min = min(p_vals)
    p_max = max(p_vals)
    s_max = max(s_vals) * 1.1  # extra margin above max measured speedup

    # Create figure
    fig, ax = plt.subplots(figsize=(10,6))

    ########################################################################
    # (Optional) Clamp x-axis if p_max is huge compared to s_max
    ########################################################################
    clamp_factor = 2.0
    if p_max > clamp_factor * s_max:
        print(f"NOTE: p_max={p_max:.2f} >> s_max={s_max:.2f}, "
              f"clamping x-axis to {clamp_factor}*s_max for clarity.")
        x_max_vis = max(p_max, clamp_factor * s_max)
    else:
        x_max_vis = p_max

    ########################################################################
    # Build a DENSE array of p-values for the stepped fill/line
    # Instead of 1 point per integer, we use e.g. 500 steps for a smoother look
    ########################################################################
    STEPS = 1000000  # Increase/decrease for finer/coarser stepping
    # Ensure we start at least at p=1 to avoid log-scale issues
    p_start = max(1, p_min)
    p_arr = np.linspace(p_start, x_max_vis, STEPS)

    ########################################################################
    # 2) Color regions with stepped fill
    #
    #   Red   : 0 <= y < 1
    #   Green : 1 <= y < x
    #   Yellow: x <= y
    #
    # We use 'step="mid"' to get a 'stepped' boundary.
    ########################################################################

    # 2a) Red region: 0 <= y < 1
    ax.fill_between(p_arr,
                    0, 1,
                    color='red', alpha=0.15,
                    label='S(p) < 1',
                    step='mid')

    # 2b) Green region: 1 <= y < x  (but do not exceed s_max)
    #    We'll fill up to min(x, s_max).
    y_top_green = np.minimum(p_arr, s_max)
    ax.fill_between(p_arr,
                    1, y_top_green,
                    where=(p_arr >= 1),   # only fill if x >= 1
                    color='green', alpha=0.15,
                    label='1 ≤ S(p) < p',
                    step='mid')

    # 2c) Yellow region: x <= y <= s_max
    ax.fill_between(p_arr,
                    p_arr, s_max,
                    where=(p_arr <= s_max),
                    color='yellow', alpha=0.15,
                    label='S(p) ≥ p',
                    step='mid')

    ########################################################################
    # 3) Plot the measured speedups as a normal line with markers
    ########################################################################
    ax.plot(p_vals, s_vals, marker='o', linestyle='-', label='Measured Speedup')

    ########################################################################
    # 4) Plot the stepped dashed red line for y = x
    #
    #    We use the same p_arr so the "step" frequency matches the fills.
    #    'drawstyle="steps-mid"' ensures the line breaks at each p_arr point.
    ########################################################################
    ax.plot(p_arr, p_arr,
            'r--',
            alpha=0.8,
            label='Ideal: y = p',
            drawstyle='steps-mid')

    ########################################################################
    # 5) Annotate each data point with speedup
    ########################################################################
    for x_val, y_val in zip(p_vals, s_vals):
        ax.annotate(f"{y_val:.2f}",
                    xy=(x_val, y_val),
                    xytext=(0,5),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9)

    ########################################################################
    # 6) Axis ranges
    ########################################################################
    ax.set_xlim(left=p_start, right=x_max_vis)
    ax.set_ylim(bottom=0, top=max(s_max,1))

    ax.set_xlabel("Threads (p)")
    ax.set_ylabel("Speedup S(p) = T(1)/T(p)")
    ax.set_title("Speedup vs. Threads (Finer Stepped Regions & Line)")

    if use_log_scale_x:
        ax.set_xscale('log')

    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(png_filename, dpi=150)
    plt.show()
    print(f"Saved stepped speedup plot as '{png_filename}'")




def plot_efficiency_numeric(efficiencies, png_filename, use_log_scale_x=False):
    """
    Plot Efficiency(p) = S(p)/p vs p on a numeric x-axis,
    labeling each point with its value.
    
    - use_log_scale_x: if True, the x-axis will be in log scale.
    """
    if not efficiencies:
        print(f"No efficiency data to plot: {png_filename}")
        return

    # Sort by thread_count
    sorted_data = sorted(efficiencies, key=lambda x: x[0])
    thread_counts = [d[0] for d in sorted_data]
    e_values = [d[1] for d in sorted_data]

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot in numeric space: p on x-axis, E(p) on y-axis
    plt.plot(thread_counts, e_values, marker='o')

    # Annotate each point with its efficiency value
    for p, e in zip(thread_counts, e_values):
        label = f"{e:.2f}"
        plt.annotate(
            label,
            xy=(p, e),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=9
        )

    # Optional: draw a horizontal line at E=1.0 for reference
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Ideal E=1.0')
    plt.legend()

    # Log scale on the x-axis if requested
    if use_log_scale_x:
        plt.xscale('log')

    plt.xlabel("Threads (P) - Numeric scale")
    plt.ylabel("Efficiency E(P) = S(P)/P")
    plt.title("Efficiency vs. P (Numeric X)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_filename, dpi=150)
    plt.show()
    print(f"Saved numeric efficiency plot as '{png_filename}'")

###############################################################################
# 4) Main program
###############################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <datafile.txt>")
        sys.exit(1)

    datafile = sys.argv[1]
    base = os.path.splitext(datafile)[0]

    # 1) Parse data
    times_by_thread = parse_gpu_important_times(datafile)

    # 2) Compute average times
    averages = compute_average_times(times_by_thread)
    print("Averages:")
    for thr, t in averages:
        print(f"  P = {thr}, T(P) = {t:.2f} µs")

    # 3) Compute speedups
    speedups = compute_speedups(averages)
    print("\nSpeedups (S(P) = T(1)/T(P)):")
    for thr, s in speedups:
        print(f"  P = {thr}, S(P) = {s:.2f}")

    # 4) Compute efficiencies
    efficiencies = compute_efficiencies(averages)
    print("\nEfficiencies (E(P) = S(P)/P):")
    for thr, e in efficiencies:
        print(f"  P = {thr}, E(P) = {e:.2f}")

    # 5) Plot T(P) with log-x
    #plot_time_numeric(
    #    averages, 
    #    f"{base}_time_logx.png",
    #    use_log_scale_x=True, 
    #    use_log_scale_y=False
    #)

    # 6) Plot T(P) with log-y (optional, just an example)
    # plot_time_numeric(
    #    averages, 
    #    f"{base}_time_logy.png",
    #    use_log_scale_x=False,
    #    use_log_scale_y=True
    # )

    # 7) Plot speedup with log-x
    plot_speedup_numeric_colored_stepped(
        speedups,
        f"{base}_speedup_logx.png",
        use_log_scale_x=True
    )

    # 8) Plot efficiency with log-x
    plot_efficiency_numeric(
        efficiencies,
        f"{base}_efficiency_logx.png",
        use_log_scale_x=True
    )
