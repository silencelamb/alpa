from ortools.sat.python import cp_model

def rectangle_areas():
    for i in range(1, 6):
        for j in range(1, 6):
            yield i, j


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        for i in range(5):
            print("\n")
            for j in range(5):
                print(f"{self.Value(self.__variables[i*5+j])}", end=" ")
        print("==========\n")

    def solution_count(self):
        return self.__solution_count
    
    
def solve_5x5_square():
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()

    total_area = 5 * 5
    rectangles = list(rectangle_areas())

    num_rects = len(rectangles)
    count_vars = [model.NewIntVar(0, total_area // (w*h), f'count_{w}x{h}') for w, h in rectangles]

    area_covered = sum([count_vars[i] * w * h for i, (w, h) in enumerate(rectangles)])
    model.Add(area_covered == total_area)

    # Solve
    status = solver.Solve(model)

    solution_printer = VarArraySolutionPrinter(count_vars)
    # Enumerate all solutions.
    solver.parameters.enumerate_all_solutions = True
    # Solve.
    status = solver.Solve(model, solution_printer)
    # status = solver.Solve(model)

    print(f"Status = {solver.StatusName(status)}")

solve_5x5_square()
