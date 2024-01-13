import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from multiprocessing import Pool

# 读取生成的栅格文件
grid = np.load('grid_data.npy')

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

        # 在二维栅格图上展示结果
        plt.imshow(grid[:, :, 0], cmap='viridis', extent=[0, grid.shape[1], 0, grid.shape[0]])
        plt.plot([x for x, y in shortest_path], [y for x, y in shortest_path], color='red', linewidth=2)
        plt.colorbar(label='Z_altitude')
        plt.xlabel('X_position')
        plt.ylabel('Y_position')
        plt.title('A* Path Planning Result')
        plt.savefig('a_star_path_plot.png', format='png', bbox_inches='tight')
        plt.show()
    else:
        print("No valid path found.")
