import numpy as np
import math
import random
import matplotlib.pyplot as plt

DATA_NUM = 1000  # 你可以根据需要调整数据量
CIRCLE_RADIUS = 1.0 # 圆的半径 (假设圆心在 (0, 0))
SQUARE_LIMIT = 2.0   # 正方形区域的边界 (-2 to 2)

def tag_entry(x, y):
    """判断点 (x, y) 是否在半径为 CIRCLE_RADIUS 的圆内"""
    if x**2 + y**2 <= CIRCLE_RADIUS**2:
        return 1  # 圆内
    else:
        return 0  # 圆外

def create_round_data(num=DATA_NUM):
    """生成平衡的数据集，圆内点和圆外点数量大致相等"""
    points_inside = []
    points_outside = []
    num_inside_desired = num // 2  # 期望的圆内点数量 (大约一半)
    num_outside_desired = num - num_inside_desired # 期望的圆外点数量 (剩余部分)

    # 生成圆内点
    while len(points_inside) < num_inside_desired:
        x = random.uniform(-CIRCLE_RADIUS, CIRCLE_RADIUS) # 在包含圆的方形内随机取 x
        y = random.uniform(-CIRCLE_RADIUS, CIRCLE_RADIUS) # 在包含圆的方形内随机取 y
        if tag_entry(x, y) == 1: # 检查是否在圆内
            points_inside.append([x, y, 1]) # 添加圆内点，标签为 1

    # 生成圆外点
    while len(points_outside) < num_outside_desired:
        x = random.uniform(-SQUARE_LIMIT, SQUARE_LIMIT) # 在正方形区域内随机取 x
        y = random.uniform(-SQUARE_LIMIT, SQUARE_LIMIT) # 在正方形区域内随机取 y
        if tag_entry(x, y) == 0: # 检查是否在圆外
            points_outside.append([x, y, 0]) # 添加圆外点，标签为 0

    # 合并圆内点和圆外点列表
    entry_list = points_inside + points_outside

    # 转换为 NumPy 数组并打乱顺序 (可选，但通常推荐)
    entry_array = np.array(entry_list)
    np.random.shuffle(entry_array) # 原地打乱数组

    return entry_array


def plot_data(data, title):
    color = []
    for i in data[:, 2]:
        if i == 0:
            color.append('orange')
        else:
            color.append('green')
    plt.title(title)
    # plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], c=color)
    plt.show()


def main():
    data = create_round_data(1000)
    plot_data(data, 'data')

if __name__ == '__main__':
    main()