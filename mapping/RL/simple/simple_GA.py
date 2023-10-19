import numpy as np
from pymoo.model.problem import Problem
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.optimize import minimize
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation

# 定义问题
class CuttingProblem(Problem):

    def __init__(self):
        super().__init__(n_var=25, n_obj=1, n_constr=0, type_var=int)
        self.xl = np.ones(25, dtype=int)     # 最小值为1
        self.xu = 6 * np.ones(25, dtype=int) # 假设最大值为6

    def _evaluate(self, X, out, *args, **kwargs):
        # 评估reward，这里你需要按照你的实际情况进行修改
        rewards = []
        for x in X:
            reward = self.calculate_reward(x)
            rewards.append(reward)
        out["F"] = -1.0 * np.array(rewards) # 由于我们希望最大化reward，而pymoo默认是最小化问题，因此我们取负值

    def calculate_reward(self, x):
        # 根据x计算reward，你需要自定义这个函数
        # 暂时假设随机reward
        return np.random.rand()

# 定义交叉和突变
class MyCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)  # parents, offspring

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.full((self.n_offsprings, n_matings, problem.n_var), -1)
        
        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]
            point = np.random.randint(0, n_var)
            Y[0, k, :point] = p1[:point]
            Y[0, k, point:] = p2[point:]
        
        return Y

class MyMutation(Mutation):

    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        
        for i in range(len(X)):
            if np.random.rand() < 0.1:  # mutation probability
                pos = np.random.randint(0, problem.n_var)
                Y[i, pos] = np.random.randint(problem.xl[pos], problem.xu[pos] + 1)
        
        return Y

# 实例化问题
problem = CuttingProblem()

# 设置遗传算法参数
algorithm = GA(
    pop_size=500,
    crossover=MyCrossover(),
    mutation=MyMutation(),
    eliminate_duplicates=True,
    n_offsprings=25,
    n_gen=1000
)

res = minimize(
    problem,
    algorithm,
    seed=1,
    verbose=True
)

print("Best solution found: \nX = ", res.X, "\nF = ", res.F)
