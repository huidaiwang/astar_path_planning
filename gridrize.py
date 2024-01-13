import pandas as pd
import numpy as np

# Read 3D point cloud data
data = pd.read_csv('132w-xyz-地形散点数据.txt', sep=' ', header=None)

# Define grid parameters
grid_size = 1  # size of each grid cell
x_range = (0, 300)
y_range = (0, 200)

# Calculate grid dimensions
x_cells = int((x_range[1] - x_range[0]) / grid_size)
y_cells = int((y_range[1] - y_range[0]) / grid_size)

# Create an empty grid
grid_data = np.zeros((x_cells, y_cells), dtype=float)

# Rasterize 3D points onto the grid
for i in range(len(data)):
    x_index = int((data.iloc[i, 0] - x_range[0]) / grid_size)
    y_index = int((data.iloc[i, 1] - y_range[0]) / grid_size)
    grid_data[x_index, y_index] = data.iloc[i, 2]  # Use Z-coordinate as grid value

# Save the grid data
np.save('grid_data.npy', grid_data)
