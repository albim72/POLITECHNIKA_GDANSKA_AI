import numpy as np

def rastrigin(x):
    A = 10
    return A*len(x) + sum([(xi*xi - A*np.cos(2*np.pi*xi)) for xi in x])

mu = 10
sigma = 0.3
pop = np.random.randn(mu, 100)

for gen in range(200):
    offspring = pop + sigma * np.random.randn(mu, 100)
    scores = np.array([rastrigin(ind) for ind in offspring])
    pop = offspring[np.argsort(scores)[:mu]]

best = pop[0]
print("Best score:", rastrigin(best))
