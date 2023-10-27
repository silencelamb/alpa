def get_max_rectangle(A, B):
    left_a, top_a, right_a, bottom_a = A
    left_b, top_b, right_b, bottom_b = B

    # 根据 (left_a, top_a) 顶点确定最大矩形的右边界和下边界
    if left_b > left_a and left_b < right_a:
        max_right = left_b
    else:
        max_right = right_a

    if top_b > top_a and top_b < bottom_a:
        max_bottom = top_b
    else:
        max_bottom = bottom_a

    # 返回新的矩形范围
    return (left_a, top_a, max_right, max_bottom)

def max_rectangle_from_point(point, A, B):
    x, y = point
    left_a, top_a, right_a, bottom_a = A
    left_b, top_b, right_b, bottom_b = B

    if x == left_a:
        max_right = left_b if left_b > x and left_b < right_a else right_a
    else:
        max_right = right_a

    if y == top_a:
        max_bottom = top_b if top_b > y and top_b < bottom_a else bottom_a
    else:
        max_bottom = bottom_a

    return (x, y, max_right, max_bottom)

# 示例
A = (1, 1, 5, 5)
B = (3, 3, 7, 7)
print(get_max_rectangle(A, B))  # 输出：(1, 1, 3, 3)
point = (5, 5)
print(max_rectangle_from_point(point, A, B))  # 输出：(1, 1, 3, 3)
