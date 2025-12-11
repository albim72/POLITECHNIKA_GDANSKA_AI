import random, textwrap

templates = [
    "lambda x: sorted(x)",
    "lambda x: sorted(x, reverse=True)",
    "lambda x: sorted(x, key=lambda v: abs(v))",
]

def fitness(code):
    f = eval(code)
    arr = [random.randint(-50,50) for _ in range(200)]
    return sum(f(arr)[i] <= f(arr)[i+1] for i in range(len(arr)-1))

pop = templates[:]
for gen in range(20):
    scored = sorted(pop, key=fitness, reverse=True)
    parents = scored[:2]
    child = random.choice(parents)
    if random.random() < 0.5:
        child = child.replace("abs", "lambda z: z*z") 
    pop.append(child)

print("Best:", max(pop, key=fitness))
