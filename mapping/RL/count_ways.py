def count_ways(n, m, memo=None):
    if n > m:
        return count_ways(m, n, memo)

    if n == 1:
        return 1

    if memo is None:
        memo = {}

    if (n, m) in memo:
        return memo[(n, m)]

    ways = 0

    # 垂直切割
    for j in range(1, m):
        ways += count_ways(n, j, memo) * count_ways(n, m - j, memo)

    # 水平切割
    for i in range(1, n):
        ways += count_ways(i, m, memo) * count_ways(n - i, m, memo)

    memo[(n, m)] = ways

    return ways

result = count_ways(5, 5)
print(result)

def split_ways(n, m, memo=None):
    if n > m:
        return split_ways(m, n, memo)

    if n == 1:
        return [([(1, i) for i in range(1, m+1)],)]

    if memo is None:
        memo = {}

    if (n, m) in memo:
        return memo[(n, m)]

    ways = []

    # 垂直切割
    for j in range(1, m):
        left_ways = split_ways(n, j, memo)
        right_ways = split_ways(n, m - j, memo)
        
        for left in left_ways:
            import pdb; pdb.set_trace()
            for right in right_ways:
                new_right = [((x[0], x[1] + j) for x in rect) for rect in right]
                ways.append(left + new_right)

    # 水平切割
    for i in range(1, n):
        top_ways = split_ways(i, m, memo)
        bottom_ways = split_ways(n - i, m, memo)
        
        for top in top_ways:
            for bottom in bottom_ways:
                new_bottom = [((x[0] + i, x[1]) for x in rect) for rect in bottom]
                ways.append(top + new_bottom)

    memo[(n, m)] = ways

    return ways

result = split_ways(5, 5)
print(len(result))

