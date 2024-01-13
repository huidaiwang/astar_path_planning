'''
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
scatter = ax.scatter(x, y, z, c=z, cmap='viridis')  # 这里将z作为颜色，可根据需要修改

# 设置坐标轴标签
ax.set_xlabel('X_position')
ax.set_ylabel('Y_position')
ax.set_zlabel('Z_altitude')

# 保存图形为矢量图(SVG)
plt.savefig('3d_scatter_plot.svg', format='svg', bbox_inches='tight')

# 显示图片路径
print("3D scatter plot image saved as '3d_scatter_plot.svg'")
'''
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata

# 读取地形散点数据文件
data = pd.read_csv('132w-xyz-地形散点数据.txt', sep=' ', header=None)

# 获取数据中的x、y、z坐标
x = data[0]
y = data[1]
z = data[2]

# 创建一个图形和3D坐标轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

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

# 保存图形为矢量图(SVG)
plt.savefig('3d_surface_plot_complete.svg', format='svg', bbox_inches='tight')

# 显示图片路径
print("3D surface plot with all points image saved as '3d_surface_plot_complete.svg'")
