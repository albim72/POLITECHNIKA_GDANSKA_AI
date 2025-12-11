import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score

def fitness(params):
    alpha, l1_ratio = params
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    return -np.mean(cross_val_score(model, X, y, cv=3))  

def mutate(params):
    return params + np.random.normal(0, 0.05, size=2)

def crossover(a, b):
    return np.array([a[0], b[1]])

pop = [np.random.rand(2) for _ in range(30)]
for gen in range(50):
    scores = np.array([fitness(ind) for ind in pop])
    parents = np.array(pop)[scores.argsort()[:10]]
    children = []
    for _ in range(20):
        p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
        child = mutate(crossover(p1, p2))
        children.append(child)
    pop = list(parents) + children

best = pop[np.argmin([fitness(ind) for ind in pop])]
print("Best:", best)
