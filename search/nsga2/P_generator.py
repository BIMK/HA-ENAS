import numpy as np


def crossover(p1, p2):
    # 要交叉的位
    crossover = np.random.randint(low=0, high=2, size=(16), dtype='int')
    for idx, flag in enumerate(crossover):
        if flag:
            p1[idx], p2[idx] = p2[idx], p1[idx]
    return mutation(p1), mutation(p2)


def mutation(p):
    mu = np.random.randint(low=0, high=16, size=(16), dtype='int')
    argmu = np.argwhere(mu == 0)
    for m in argmu:
        p[m] = np.random.randint(low=0, high=2, size=(1), dtype='int')
    return p


# 返回一个和交配池一样大的子代
def P_generator(MatingPool):
    N, D = MatingPool.shape
    Offspring = np.zeros((N, D), dtype='int')
    for i in range(0, N, 2):
        Offspring[i, :], Offspring[i + 1, :] = crossover(MatingPool[i], MatingPool[i + 1])

    return Offspring
