import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取地形散点数据文件
data = pd.read_csv('132w-xyz-地形散点数据.txt', sep=' ', header=None)

# 定义栅格尺寸和范围
grid_size = 1
x_range = (0, 300)
y_range = (0, 200)

# 计算栅格数量
num_x = int((x_range[1] - x_range[0]) / grid_size) + 1
num_y = int((y_range[1] - y_range[0]) / grid_size) + 1

# 创建空的二维栅格数组
grid = np.zeros((num_y, num_x, 1))  # 只记录自身高程

# 将点云数据映射到二维栅格
for i in range(len(data)):
    x_index = int((data.iloc[i, 0] - x_range[0]) / grid_size)
    y_index = int((data.iloc[i, 1] - y_range[0]) / grid_size)

    # Ensure that the indices are within bounds
    x_index = max(0, min(x_index, num_x - 1))
    y_index = max(0, min(y_index, num_y - 1))

    # 记录自身高程
    grid[y_index, x_index, 0] = data.iloc[i, 2]

# 保存结果为文件
np.save('grid_data.npy', grid)

# 绘制二维栅格图
plt.imshow(grid[:, :, 0], cmap='viridis', extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
plt.colorbar(label='Z_altitude')
plt.xlabel('X_position')
plt.ylabel('Y_position')
plt.title('2D Grid Plot')
plt.savefig('2d_grid_plot.png', format='png', bbox_inches='tight')
plt.show()
