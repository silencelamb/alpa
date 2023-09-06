def can_place(board, row, col, height, width):
    if row + height > 5 or col + width > 5:
        return False
    for i in range(row, row + height):
        for j in range(col, col + width):
            if board[i][j]:
                return False
    return True

def place(board, row, col, height, width, val=True):
    for i in range(row, row + height):
        for j in range(col, col + width):
            board[i][j] = val

memo = {}  # For memoization

def backtrack(board):
    key = tuple(tuple(row) for row in board)
    if key in memo:
        return memo[key]

    for i in range(5):
        for j in range(5):
            if not board[i][j]:
                # Try placing larger rectangles first
                for height in range(5, 0, -1):
                    for width in range(5, 0, -1):
                        if can_place(board, i, j, height, width):
                            place(board, i, j, height, width)
                            if backtrack(board):
                                memo[key] = True
                                return True
                            # Remove the rectangle (backtrack)
                            place(board, i, j, height, width, False)
                memo[key] = False
                return False  # We can't place a valid rectangle for this cell

    memo[key] = True
    return True  # The board is full

def count_solutions():
    board = [[False for _ in range(5)] for _ in range(5)]
    solutions = []

    def solve(board):
        for i in range(5):
            for j in range(5):
                if not board[i][j]:
                    for height in range(5, 0, -1):
                        for width in range(5, 0, -1):
                            if can_place(board, i, j, height, width):
                                place(board, i, j, height, width)
                                new_board = [row.copy() for row in board]
                                if new_board not in solutions:
                                    if backtrack(new_board):
                                        solutions.append(new_board)
                                    solve(board)
                                place(board, i, j, height, width, False)
                    return

    solve(board)
    return len(solutions)

print(count_solutions())
