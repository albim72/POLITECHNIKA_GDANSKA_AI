import math
import random

def f(x):
    return -x**2 + 4*x

def simulated_annealing(f, x_min, x_max, max_iter=1000, start_temp=100, alpha=0.95):
    current_x = random.uniform(x_min, x_max)
    current_value = f(current_x)
    best_x = current_x
    best_value = current_value
    temp = start_temp

    for i in range(max_iter):
        # Nowe rozwiązanie w pobliżu
        next_x = current_x + random.uniform(-0.1, 0.1)
        next_x = max(x_min, min(x_max, next_x))  # ograniczenie do zakresu
        next_value = f(next_x)

        delta = next_value - current_value

        # Akceptuj lepsze lub (czasem) gorsze rozwiązania
        if delta > 0 or random.random() < math.exp(delta / temp):
            current_x = next_x
            current_value = next_value

        # Aktualizuj najlepsze
        if current_value > best_value:
            best_x = current_x
            best_value = current_value

        temp *= alpha  # zmniejsz temperaturę

    return best_x, best_value

# Uruchomienie
x, fx = simulated_annealing(f, 0, 4)
print(f"Maksimum funkcji w x = {x:.4f}, f(x) = {fx:.4f}")
