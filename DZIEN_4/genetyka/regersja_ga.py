import operator, math, random
import numpy as np

# ====== Dane docelowe: y = sin(x) + x^2 ======
X = np.linspace(-2, 2, 100)
y = np.sin(X) + X**2

# ====== Zestaw operacji (alfabet "wzoru") ======
ops = [
    ("add", operator.add, 2),
    ("mul", operator.mul, 2),
    ("sin", math.sin, 1)
]

# pomocniczo: operacje pogrupowane po arności
ops_by_arity = {
    1: [o for o in ops if o[2] == 1],
    2: [o for o in ops if o[2] == 2],
}


# ====== Reprezentacja drzewa i generator losowych drzew ======
# Drzewo ma postać:
#  - ("var", None)           → liść, zmienna x
#  - ("add", [left, right])  → (left + right)
#  - ("mul", [left, right])  → (left * right)
#  - ("sin", [child])        → sin(child)

def random_tree(depth=3):
    """Losowo generuje drzewo o maksymalnej głębokości 'depth'."""
    if depth == 0 or random.random() < 0.3:
        return ("var", None)
    op_name, op_fun, arity = random.choice(ops)
    return (op_name, [random_tree(depth - 1) for _ in range(arity)])


def eval_tree(tree, x):
    """Oblicza wartość drzewa dla danej wartości x."""
    name, children = tree
    if name == "var":
        return x
    op_name, op_fun, arity = [o for o in ops if o[0] == name][0]
    vals = [eval_tree(c, x) for c in children]
    return op_fun(*vals)


def mse(tree):
    """Średni błąd kwadratowy na danych (MSE)."""
    try:
        preds = np.array([eval_tree(tree, float(xi)) for xi in X])
        # jeśli coś eksploduje numerycznie, traktujemy jako bardzo zły osobnik
        if np.any(~np.isfinite(preds)):
            return 1e9
        return float(np.mean((preds - y) ** 2))
    except Exception:
        return 1e9


def clone_tree(tree):
    """Głęboka kopia drzewa (bez współdzielenia referencji)."""
    name, children = tree
    if children is None:
        return (name, None)
    return (name, [clone_tree(c) for c in children])


# ====== Mutacja drzewa ======
def mutate(tree, depth=3, p_subtree=0.1, p_change_op=0.2):
    """
    Mutacja:
      - z pewnym prawdopodobieństwem podmieniamy całe poddrzewo na nowe,
      - rekurencyjnie mutujemy dzieci,
      - czasem zmieniamy operator na inny o tej samej arności.
    Zwracamy NOWE drzewo (funkcyjnie).
    """
    if depth <= 0:
        return ("var", None)

    name, children = tree

    # 1) z pewnym prawdopodobieństwem wymieniamy całe poddrzewo
    if random.random() < p_subtree:
        return random_tree(depth)

    # 2) liść "var" – czasem pozwalamy mu urosnąć
    if name == "var":
        if random.random() < 0.1:
            return random_tree(1)
        return ("var", None)

    # 3) mutacja w dzieciach
    new_children = [mutate(c, depth - 1, p_subtree, p_change_op) for c in children]

    # 4) czasem zmieniamy operator (ale trzymamy arność)
    if random.random() < p_change_op:
        arity = len(new_children)
        op_name, op_fun, _ = random.choice(ops_by_arity[arity])
        return (op_name, new_children)

    return (name, new_children)


# ====== Pretty-print: drzewo → wzór jako string ======
def tree_to_str(tree):
    name, children = tree
    if name == "var":
        return "x"
    if name == "sin":
        return f"sin({tree_to_str(children[0])})"
    if name == "add":
        return f"({tree_to_str(children[0])} + {tree_to_str(children[1])})"
    if name == "mul":
        return f"({tree_to_str(children[0])} * {tree_to_str(children[1])})"
    return "?"


# ====== Pętla ewolucyjna ======
random.seed(42)

pop_size = 50
pop = [random_tree(depth=4) for _ in range(pop_size)]

def best_of_pop(pop):
    fits = [mse(t) for t in pop]
    idx = int(np.argmin(fits))
    return pop[idx], fits[idx]


for gen in range(40):
    # ocena całej populacji
    fits = [mse(t) for t in pop]
    order = np.argsort(fits)
    pop = [pop[i] for i in order]
    fits = [fits[i] for i in order]

    best_tree, best_fit = pop[0], fits[0]
    if gen % 5 == 0 or gen == 39:
        print(f"Gen {gen:02d}: best MSE={best_fit:.4e}, expr={tree_to_str(best_tree)}")

    # elita: top 10
    elites = pop[:10]

    # dzieci: mutacje losowo wybieranych elit
    children = []
    while len(children) < pop_size - len(elites):
        parent = clone_tree(random.choice(elites))
        child = mutate(parent, depth=4)
        children.append(child)

    pop = elites + children

# wynik końcowy
best_tree, best_fit = best_of_pop(pop)
print("\nFinal best MSE:", best_fit)
print("Expression:", tree_to_str(best_tree))
