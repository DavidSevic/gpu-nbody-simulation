import plotly.graph_objects as go

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
z = [row[4] for row in data]

fig = go.Figure()

unique_bodies = sorted(set(body))
for b in unique_bodies:
    indices = [i for i, value in enumerate(body) if value == b]
    body_x = [x[i] for i in indices]
    body_y = [y[i] for i in indices]
    body_z = [z[i] for i in indices]

    fig.add_trace(go.Scatter3d(
        x=body_x, 
        y=body_y, 
        z=body_z,
        mode='lines+markers',
        marker=dict(
            size=5, 
            color=time,
            colorscale='Viridis',
        ),
        name=f'Body {b}'
    ))

fig.update_layout(
    title='3D N-Body Problem Visualization',
    scene=dict(
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        zaxis_title='Z Coordinate',
    ),
    showlegend=True
)

plt.savefig('plot_3d.png')
fig.show()
