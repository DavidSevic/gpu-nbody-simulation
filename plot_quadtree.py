#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt
import sys

occupant_pattern = re.compile(
    r"occupantIndex=(-?\d+)\s+occupantPos=\(([-0-9.e+]+),([-0-9.e+]+)\)"
)

def parse_quadtree_file(filename):
    """
    Reads each line of 'filename', which should contain:
      depth  X_MIN  X_MAX  Y_MIN  Y_MAX  total_mass [ occupantIndex=... occupantPos=(...,...) ... ]
    Returns a list of tuples:
      (depth, x_min, x_max, y_min, y_max, total_mass, [(occIdx, occX, occY), ...])
    """
    results = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            tokens = line.split()
            if len(tokens) < 6:
                continue

            depth      = int(tokens[0])
            x_min      = float(tokens[1])
            x_max      = float(tokens[2])
            y_min      = float(tokens[3])
            y_max      = float(tokens[4])
            total_mass = float(tokens[5])

            occupantMatches = occupant_pattern.findall(line)
            occupantPositions = []
            for match in occupantMatches:
                occIndex = int(match[0])
                occX = float(match[1])
                occY = float(match[2])
                occupantPositions.append((occIndex, occX, occY))

            results.append((depth, x_min, x_max, y_min, y_max, total_mass, occupantPositions))
    return results

def plot_quadtree(quadtree_data):
    """
    Plots each node's bounding box (rectangle).
    Also plots occupant positions as red dots.
    """
    fig, ax = plt.subplots()
    
    occupant_xs = []
    occupant_ys = []

    for entry in quadtree_data:
        depth, x_min, x_max, y_min, y_max, total_mass, occupant_list = entry
        width  = x_max - x_min
        height = y_max - y_min

        rect = plt.Rectangle(
            (x_min, y_min), width, height,
            fill=False, 
            edgecolor='black',
            alpha=0.3
        )
        ax.add_patch(rect)

        for (occIndex, occX, occY) in occupant_list:
            occupant_xs.append(occX)
            occupant_ys.append(occY)

    ax.scatter(occupant_xs, occupant_ys, color='red', s=20, zorder=3)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.title("Barnes-Hut Quadtree Visualization")
    plt.show()

if __name__ == "__main__":
    filename = sys.argv[1]#"quadtree.txt"

    data = parse_quadtree_file(filename)

    plot_quadtree(data)
