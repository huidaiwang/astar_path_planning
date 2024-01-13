import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata
from queue import PriorityQueue
from multiprocessing import Pool

# 读取地形散点数据文件
data = pd.read_csv('132w-xyz-地形散点数据.txt', sep=' ', header=None)

# 创建一个图形和3D坐标轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 获取数据中的x、y、z坐标
x = data[0]
y = data[1]
z = data[2]

# 绘制三维散点图
scatter = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')  # 这里将z作为颜色，可根据需要修改

# 创建网格
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# 进行网格内插，生成曲面
zi = griddata((x, y), z, (xi, yi), method='linear')

# 绘制曲面
surf = ax.plot_surface(xi, yi, zi, alpha=0.7, cmap='viridis', edgecolors='k', rstride=5, cstride=5)

# 设置坐标轴标签
ax.set_xlabel('X_position')
ax.set_ylabel('Y_position')
ax.set_zlabel('Z_altitude')

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
# np.save('grid_data.npy', grid)

# 定义路径规划的消耗值
COST_STRAIGHT = 1
COST_UPHILL = 2
COST_CONTINUOUS_UPHILL = 2.5
COST_DIRECTION_CHANGE = 1.3
COST_DIAGONAL = 1.4

# 定义起始点和目标点
start_point = (0, 0)  # 栅格左下角
end_point = (grid.shape[1] - 1, grid.shape[0] - 1)  # 栅格右上角

# 定义A星路径规划算法
def astar(grid, start, end):
    def heuristic(a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def get_neighbors(current):
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                new_x, new_y = current[0] + i, current[1] + j
                if 0 <= new_x < grid.shape[1] and 0 <= new_y < grid.shape[0]:
                    neighbors.append((new_x, new_y))
        return neighbors

    def calculate_cost(current, neighbor):
        dx = abs(current[0] - neighbor[0])
        dy = abs(current[1] - neighbor[1])
        if dx + dy == 2:  # Diagonal movement
            return COST_DIAGONAL
        elif dx + dy == 1:  # Straight movement
            return COST_STRAIGHT
        else:
            return COST_DIRECTION_CHANGE

    open_set = PriorityQueue()
    open_set.put((0, start))

    came_from = {}
    cost_so_far = {start: 0}

    while not open_set.empty():
        current = open_set.get()[1]

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for neighbor in get_neighbors(current):
            new_cost = cost_so_far[current] + calculate_cost(current, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(end, neighbor)
                open_set.put((priority, neighbor))
                came_from[neighbor] = current

    return []

# 多线程执行A星路径规划
def parallel_astar(start, end):
    path = astar(grid[:, :, 0], start, end)
    return path

if __name__ == '__main__':
    # 利用多线程执行A星路径规划
    pool = Pool()
    results = [pool.apply_async(parallel_astar, args=(start_point, end_point)) for _ in range(4)]  # 使用4个线程

    paths = [result.get() for result in results if result.get()]

    if paths:
        # 选择最短路径
        shortest_path = min(paths, key=len)

        # 在三维地形图表层上显示最优路径
        path_x = [point[0] * grid_size + x_range[0] for point in shortest_path]
        path_y = [point[1] * grid_size + y_range[0] for point in shortest_path]
        path_z = [grid[point[1], point[0], 0] for point in shortest_path]

        ax.plot(path_x, path_y, path_z, c='red', linewidth=3)

        # 保存图形为矢量图(SVG)
        plt.savefig('3d_path_plot.svg', format='svg', bbox_inches='tight')
        print("3D path plot image saved as '3d_path_plot.svg'")
    else:
        print("No valid path found.")
