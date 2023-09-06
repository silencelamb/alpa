import matplotlib.pyplot as plt
import numpy as np

def plot_grid(actions, title):
    plt.figure(figsize=(10, 10))

    # 定义节点位置
    positions = {
        1: (1, 3),
        2: (2, 3),
        3: (3, 3),
        4: (1, 2),
        5: (2, 2),
        6: (3, 2),
        7: (1, 1),
        8: (2, 1),
        9: (3, 1)
    }

    # 画出节点
    for node, (x, y) in positions.items():
        plt.scatter(x, y, s=100, color='blue', zorder=5)
        plt.text(x, y, str(node), ha='center', va='center', fontsize=15, color='white', zorder=5)

    # 画出每个节点的发送和接收动作
    for node, neighbors in actions.items():
        for neighbor in neighbors:
            plt.arrow(positions[node][0], positions[node][1], 
                      positions[neighbor][0] - positions[node][0], 
                      positions[neighbor][1] - positions[node][1], 
                      head_width=0.1, head_length=0.1, fc='red', ec='red', zorder=0)

    plt.title(title)
    plt.axis('off')
    plt.show()

# 定义每个时间节拍的动作
actions_t1 = {
    1: [2, 4],
    2: [1, 3, 5],
    3: [2, 6],
    4: [1, 5, 7],
    5: [2, 4, 6, 8],
    6: [3, 5, 9],
    7: [4, 8],
    8: [5, 7, 9],
    9: [6, 8]
}

actions_t2 = {
    1: [2, 4],
    2: [1, 3, 5],
    3: [2, 6],
    4: [1, 5, 7],
    5: [2, 4, 6, 8],
    6: [3, 5, 9],
    7: [4, 8],
    8: [5, 7, 9],
    9: [6, 8]
}

# (其他时间节拍的动作可以类似地定义)

plot_grid(actions_t1, "时间节拍1")
plot_grid(actions_t2, "时间节拍2")
# 对于其他的时间节拍，您可以按照类似的方法来绘制

