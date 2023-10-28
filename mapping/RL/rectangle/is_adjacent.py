def is_adjacent_no_overlap(rect1, rect2):
    col_start1, row_start1, col_end1, row_end1 = rect1
    col_start2, row_start2, col_end2, row_end2 = rect2
    # 检查是否在水平方向相邻
    horizontal_adjacent = (
        (col_end1+1 == col_start2 and not (row_start1 > row_end2 or row_end1 < row_start2)) or
        (col_start1 == col_end2+1 and not (row_start1 > row_end2 or row_end1 < row_start2))
    )

    # 检查是否在垂直方向相邻
    vertical_adjacent = (
        (row_end1+1 == row_start2 and not (col_start1 > col_end2 or col_end1 < col_start2)) or
        (row_start1 == row_end2+1 and not (col_start1 > col_end2 or col_end1 < col_start2))
    )

    return horizontal_adjacent or vertical_adjacent

# 测试
rect1 = (0, 0, 2, 2)
rect2 = (3, 0, 4, 2)
print(is_adjacent_no_overlap(rect1, rect2))  # 输出 True，因为两矩形在水平方向上相邻

rect1 = (3, 2, 3, 3)
rect2 = [0, 0, 3, 1]
print(is_adjacent_no_overlap(rect1, rect2))