import numpy as np

print("========****1****==========")
def is_overlapping(x1, y1, x2, y2, i, j, p, q):
    return not (q < x1 or i > x2 or p < y1 or j > y2)

def enumerate_w(m, n, x1, y1, x2, y2):
    for i in range(m):
        for j in range(n):
            for p in range(i, m):
                for q in range(j, n):
                    if not is_overlapping(x1, y1, x2, y2, i, j, p, q):
                        print((i, j, p, q))

# 测试
m, n = 5, 5
x1, y1, x2, y2 = 1, 1, 3, 3
enumerate_w(m, n, x1, y1, x2, y2)

print("========****2****==========")

def enumerate_w_vectorized(m, n, x1, y1, x2, y2):
    # 初始化矩阵为1
    matrix = np.ones((m, n), dtype=int)
    
    # 将矩形 s 的位置设置为0
    matrix[x1:x2+1, y1:y2+1] = 0
    
    # 遍历所有可能的矩形 w 的高和宽
    for height in range(1, m + 1):
        for width in range(1, n + 1):
            # 创建一个卷积核
            kernel = np.ones((height, width), dtype=int)
            
            # 使用卷积检查每个位置
            conv = np.where(np.equal(kernel.sum(), matrix[x1:x1+height, y1:y1+width].sum()))

            for x, y in zip(*conv):
                if matrix[x:x+height, y:y+width].sum() == height * width:
                    print((x, y, x + height - 1, y + width - 1))

# 测试
m, n = 5, 5
x1, y1, x2, y2 = 1, 1, 3, 3
enumerate_w_vectorized(m, n, x1, y1, x2, y2)

print("========****3****==========")

import numpy as np

def is_overlapping_with_prefix_sum(prefix_sum, x1, y1, x2, y2):
    total = prefix_sum[x2][y2]
    total -= prefix_sum[x1-1][y2] if x1 > 0 else 0
    total -= prefix_sum[x2][y1-1] if y1 > 0 else 0
    total += prefix_sum[x1-1][y1-1] if x1 > 0 and y1 > 0 else 0
    return total != (x2 - x1 + 1) * (y2 - y1 + 1)

def enumerate_w_with_prefix_sum(m, n, x1, y1, x2, y2):
    matrix = np.ones((m, n), dtype=int)
    matrix[x1:x2+1, y1:y2+1] = 0
    
    prefix_sum = np.cumsum(np.cumsum(matrix, axis=0), axis=1)
    
    for i in range(m):
        for j in range(n):
            for p in range(i, m):
                for q in range(j, n):
                    if not is_overlapping_with_prefix_sum(prefix_sum, i, j, p, q):
                        print((i, j, p, q))

# 测试
m, n = 5, 5
x1, y1, x2, y2 = 1, 1, 3, 3
enumerate_w_with_prefix_sum(m, n, x1, y1, x2, y2)

print("========****4****==========")

import numpy as np

def enumerate_w_vectorized(m, n, x1, y1, x2, y2):
    # 对于 w 的左上角和右下角，生成坐标网格
    i, j = np.arange(m)[:, None, None, None], np.arange(n)[:, None, None]
    p, q = np.arange(m)[None, :, None, None], np.arange(n)[None, None, :, None]
    
    # 使用广播技术检查每个可能的 w 是否与 s 重叠
    no_overlap = np.logical_or.reduce((
        q < x1,
        i > x2,
        p < y1,
        j > y2
    ))
    
    # 根据上述条件获取 w 的坐标
    w_coords = np.argwhere(no_overlap)
    
    for coord in w_coords:
        print(tuple(coord))

# 测试
m, n = 5, 5
x1, y1, x2, y2 = 1, 1, 3, 3
enumerate_w_vectorized(m, n, x1, y1, x2, y2)

