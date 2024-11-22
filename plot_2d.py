import matplotlib.pyplot as plt

file_path = 'positions.txt'
data = []

with open(file_path, 'r') as file:
    for line in file:
        values = list(map(float, line.split()))
        data.append(values)

time = [row[0] for row in data]
body = [int(row[1]) for row in data]
x = [row[2] for row in data]
y = [row[3] for row in data]

plt.figure(figsize=(8, 8))

unique_bodies = sorted(set(body))
for b in unique_bodies:
    indices = [i for i, value in enumerate(body) if value == b]
    body_x = [x[i] for i in indices]
    body_y = [y[i] for i in indices]
    plt.plot(body_x, body_y, marker='o', label=f'Body {b}')

plt.title("N-Body Problem Visualization")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.savefig('plot_2d.png')
plt.show()
