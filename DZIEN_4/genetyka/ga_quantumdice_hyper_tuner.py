
"""
ga_quantumdice_hyper_tuner.py

Genetic Algorithm hyperparameter tuner for a simple MLP classifier,
extended with a QuantumDice-style meta-controller that adaptively
narrows the neuron range in regions of the hyperparameter space.

Tested with TensorFlow 2.x (e.g. in Google Colab).
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


# ============================================================
# 1. Dataset utilities (MNIST, binary or multi-class)
# ============================================================

def load_mnist_binary():
    """
    Ładuje MNIST i zamienia na prosty problem binarnej klasyfikacji:
    cyfry 0–4 -> klasa 0, 5–9 -> klasa 1.
    Zwraca X_train, X_val, y_train, y_val (one-hot).
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    # binarna etykieta
    y_bin = (y >= 5).astype(np.int32)

    # normalizacja i spłaszczenie
    x = x.astype("float32") / 255.0
    x = x.reshape((x.shape[0], -1))

    X_train, X_val, y_train_raw, y_val_raw = train_test_split(
        x, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )

    num_classes = 2
    y_train = to_categorical(y_train_raw, num_classes)
    y_val = to_categorical(y_val_raw, num_classes)

    return X_train, X_val, y_train, y_val, x.shape[1], num_classes


# ============================================================
# 2. Chromosome representation & model building
# ============================================================

@dataclass
class Chromosome:
    """Prosty chromosom z hiperparametrami MLP."""
    num_layers: int
    num_neurons: int
    learning_rate: float


def build_model_from_chromosome(
    chrom: Chromosome,
    input_dim: int,
    num_classes: int
):
    """Buduje model Keras MLP na podstawie chromosomu."""
    model = Sequential()
    model.add(Dense(chrom.num_neurons, activation="relu", input_shape=(input_dim,)))
    for _ in range(chrom.num_layers - 1):
        model.add(Dense(chrom.num_neurons, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    opt = Adam(learning_rate=chrom.learning_rate)
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================================================
# 3. Genetic Algorithm core
# ============================================================

def random_chromosome(
    layers_range: Tuple[int, int],
    neurons_range: Tuple[int, int],
    learning_rates: List[float],
    rng: random.Random,
) -> Chromosome:
    min_L, max_L = layers_range
    min_N, max_N = neurons_range
    num_layers = rng.randint(min_L, max_L)
    num_neurons = rng.randint(min_N, max_N)
    lr = rng.choice(learning_rates)
    return Chromosome(num_layers, num_neurons, lr)


def mutate(
    chrom: Chromosome,
    layers_range: Tuple[int, int],
    neurons_range: Tuple[int, int],
    learning_rates: List[float],
    mutation_rate: float,
    rng: random.Random,
) -> Chromosome:
    min_L, max_L = layers_range
    min_N, max_N = neurons_range

    num_layers = chrom.num_layers
    num_neurons = chrom.num_neurons
    lr = chrom.learning_rate

    if rng.random() < mutation_rate:
        num_layers = rng.randint(min_L, max_L)
    if rng.random() < mutation_rate:
        num_neurons = rng.randint(min_N, max_N)
    if rng.random() < mutation_rate:
        lr = rng.choice(learning_rates)

    return Chromosome(num_layers, num_neurons, lr)


def crossover(parent1: Chromosome, parent2: Chromosome, rng: random.Random) -> Chromosome:
    """Prosty crossover: miesza pola rodziców."""
    num_layers = parent1.num_layers if rng.random() < 0.5 else parent2.num_layers
    num_neurons = parent1.num_neurons if rng.random() < 0.5 else parent2.num_neurons
    lr = parent1.learning_rate if rng.random() < 0.5 else parent2.learning_rate
    return Chromosome(num_layers, num_neurons, lr)


def evaluate_fitness(
    chrom: Chromosome,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    num_classes: int,
    epochs: int = 3,
    batch_size: int = 128,
    verbose: int = 0,
) -> float:
    """
    Trenuje mini-MLP z zadanymi hiperparametrami przez kilka epok
    i zwraca accuracy na walidacji jako fitness.
    """
    model = build_model_from_chromosome(chrom, input_dim, num_classes)
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_data=(X_val, y_val),
    )
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    return float(val_acc)


def tournament_selection(population: List[Chromosome], fitnesses: List[float], k: int, rng: random.Random) -> Chromosome:
    """Turniejowy wybór rodzica."""
    selected_indices = [rng.randrange(len(population)) for _ in range(k)]
    best_idx = max(selected_indices, key=lambda i: fitnesses[i])
    return population[best_idx]


def genetic_algorithm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    num_classes: int,
    pop_size: int,
    generations: int,
    layers_range: Tuple[int, int],
    neurons_range: Tuple[int, int],
    learning_rates: List[float],
    mutation_rate: float = 0.3,
    tournament_k: int = 3,
    seed: int = 42,
) -> Tuple[Chromosome, float]:
    """
    Klasyczny GA do strojenia podstawowych hiperparametrów MLP:
      - liczba warstw,
      - liczba neuronów w warstwie,
      - learning rate.
    """
    rng = random.Random(seed)

    # Inicjalizacja populacji
    population = [
        random_chromosome(layers_range, neurons_range, learning_rates, rng)
        for _ in range(pop_size)
    ]

    fitnesses = [
        evaluate_fitness(chrom, X_train, y_train, X_val, y_val, input_dim, num_classes)
        for chrom in population
    ]

    best_idx = int(np.argmax(fitnesses))
    best_chrom = population[best_idx]
    best_fit = fitnesses[best_idx]

    print(f"  [GA] Init best fitness: {best_fit:.4f} "
          f"(layers={best_chrom.num_layers}, neurons={best_chrom.num_neurons}, lr={best_chrom.learning_rate})")

    # Główna pętla GA
    for gen in range(1, generations + 1):
        new_population = []

        # elityzm: kopiujemy najlepszego bez zmian
        new_population.append(best_chrom)

        # reszta populacji
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, tournament_k, rng)
            parent2 = tournament_selection(population, fitnesses, tournament_k, rng)
            child = crossover(parent1, parent2, rng)
            child = mutate(child, layers_range, neurons_range, learning_rates, mutation_rate, rng)
            new_population.append(child)

        # ocena nowej populacji
        new_fitnesses = [
            evaluate_fitness(chrom, X_train, y_train, X_val, y_val, input_dim, num_classes)
            for chrom in new_population
        ]

        # aktualizacja najlepszego
        current_best_idx = int(np.argmax(new_fitnesses))
        current_best_chrom = new_population[current_best_idx]
        current_best_fit = new_fitnesses[current_best_idx]

        if current_best_fit > best_fit:
            best_fit = current_best_fit
            best_chrom = current_best_chrom

        population = new_population
        fitnesses = new_fitnesses

        print(f"  [GA] Gen {gen:02d}: best fitness={best_fit:.4f} "
              f"(layers={best_chrom.num_layers}, neurons={best_chrom.num_neurons}, lr={best_chrom.learning_rate})")

    return best_chrom, best_fit


# ============================================================
# 4. QuantumDice manager with adaptive neuron window
# ============================================================

class QuantumDiceManagerGAAdaptive:
    """
    Menedżer 'kwantowej kostki' nad regionami hiperparametrów:
    - każdy region ma:
        name
        layers_range = (min_layers, max_layers)
        neurons_range = (global_min_neurons, global_max_neurons)   # twarde granice
        current_neurons_range = (cur_min, cur_max)                 # aktualne, dynamiczne okno
        learning_rates
        best_fitness
        best_neurons               # ile neuronów miała najlepsza sieć w tym regionie
    - przy poprawie:
        * aktualizujemy best_fitness,
        * zawężamy current_neurons_range wokół best_neurons.
    """

    def __init__(
        self,
        regions: List[Dict[str, Any]],
        beta: float = 10.0,
        explore_eps: float = 0.1,
        shrink_factor: float = 0.5,
        min_window_size: int = 8,
        rng: random.Random | None = None,
    ):
        """
        regions: lista słowników:
          {
            "name": ...,
            "layers_range": (min_L, max_L),
            "neurons_range": (min_N, max_N),
            "learning_rates": [...]
          }

        shrink_factor: jak bardzo zwężać okno neuronów przy poprawie (0.5 = o połowę)
        min_window_size: minimalna szerokość okna neuronów (żeby się nie zwinęło do punktu)
        """
        self.rng = rng or random.Random()
        self.beta = beta
        self.explore_eps = explore_eps
        self.shrink_factor = shrink_factor
        self.min_window_size = min_window_size

        self.regions: List[Dict[str, Any]] = []
        for r in regions:
            base_min_n, base_max_n = r["neurons_range"]
            region = {
                "name": r["name"],
                "layers_range": r["layers_range"],
                "neurons_range": (base_min_n, base_max_n),           # twarde granice
                "current_neurons_range": (base_min_n, base_max_n),   # dynamiczne okno
                "learning_rates": r["learning_rates"],
                "best_fitness": 0.0,
                "best_neurons": None,
            }
            self.regions.append(region)

    # ------ rozkład po regionach ------

    def _probabilities(self) -> List[float]:
        """Softmax po best_fitness + domieszka eksploracji."""
        vals = [r["best_fitness"] for r in self.regions]
        max_v = max(vals) if vals else 0.0
        exps = [math.exp(self.beta * (v - max_v)) for v in vals]
        s = sum(exps)
        if s == 0:
            base_probs = [1.0 / len(self.regions)] * len(self.regions)
        else:
            base_probs = [e / s for e in exps]

        n = len(self.regions)
        uniform = [1.0 / n] * n
        probs = [
            (1.0 - self.explore_eps) * p + self.explore_eps * u
            for p, u in zip(base_probs, uniform)
        ]
        s2 = sum(probs)
        probs = [p / s2 for p in probs]
        return probs

    def choose_region_index(self) -> int:
        probs = self._probabilities()
        r = self.rng.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                return i
        return len(probs) - 1

    # ------ aktualizacja po runie GA ------

    def update_region(self, idx: int, new_best_fitness: float, new_best_neurons: int):
        """
        Aktualizuje region:
        - jeśli nowy wynik lepszy niż poprzedni best_fitness
          => zapisuje best_fitness, best_neurons,
             zawęża current_neurons_range wokół best_neurons.
        """
        region = self.regions[idx]
        improved = new_best_fitness > region["best_fitness"]

        if improved:
            region["best_fitness"] = float(new_best_fitness)
            region["best_neurons"] = int(new_best_neurons)

            global_min_n, global_max_n = region["neurons_range"]
            cur_min, cur_max = region["current_neurons_range"]

            # aktualna szerokość okna
            cur_width = cur_max - cur_min
            if cur_width < self.min_window_size:
                cur_width = self.min_window_size

            # nowa szerokość okna
            new_width = max(self.min_window_size, int(cur_width * self.shrink_factor))

            center_n = new_best_neurons
            new_min = max(global_min_n, center_n - new_width // 2)
            new_max = min(global_max_n, center_n + new_width // 2)

            # zabezpieczenie, żeby się nie zwinęło do jednego punktu
            if new_max <= new_min:
                new_max = min(global_max_n, new_min + self.min_window_size)

            region["current_neurons_range"] = (new_min, new_max)

        self.regions[idx] = region

    def best_region_overall(self) -> Tuple[int, Dict[str, Any]]:
        best_idx = None
        best_fit = -1.0
        for i, r in enumerate(self.regions):
            if r["best_fitness"] > best_fit:
                best_fit = r["best_fitness"]
                best_idx = i
        return best_idx, self.regions[best_idx]


# ============================================================
# 5. Helper: extract "num_neurons" from chromosome
# ============================================================

def extract_neurons_from_solution(solution: Chromosome) -> int:
    """
    Dla aktualnej reprezentacji chromosomu po prostu zwracamy num_neurons.
    Jeśli kiedyś zmienisz reprezentację na bardziej złożoną (lista warstw),
    to tutaj możesz np. zwracać max(layers) albo sum(layers).
    """
    return solution.num_neurons


# ============================================================
# 6. Meta-algorytm: QuantumDice + GA z adaptacją neuronów
# ============================================================

def quantumdice_hyper_ga_adaptive_neurons(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    num_classes: int,
    base_pop_size: int,
    base_generations: int,
    regions_config: List[Dict[str, Any]],
    meta_iters: int = 6,
) -> Tuple[Chromosome, float, Dict[str, Any]]:
    """
    Meta-algorytm:
      - QuantumDice wybiera region,
      - używamy jego current_neurons_range do ustawień GA,
      - po runie:
          * aktualizujemy best_fitness regionu,
          * zawężamy current_neurons_range wokół best_neurons.

    Zwraca:
      - global_best_solution (Chromosome),
      - global_best_fitness,
      - global_best_region (dict).
    """
    qd = QuantumDiceManagerGAAdaptive(
        regions=regions_config,
        beta=10.0,
        explore_eps=0.15,
        shrink_factor=0.5,   # zawężanie okna neuronów o połowę przy poprawie
        min_window_size=8,
    )

    global_best_solution: Chromosome | None = None
    global_best_fitness: float = 0.0
    global_best_region: Dict[str, Any] | None = None

    for meta in range(meta_iters):
        idx = qd.choose_region_index()
        region = qd.regions[idx]

        min_L, max_L = region["layers_range"]
        cur_min_n, cur_max_n = region["current_neurons_range"]
        learning_rates = region["learning_rates"]

        print(f"\n[Meta iter {meta+1}/{meta_iters}] Region #{idx} ({region['name']}):")
        print(f"  layers_range = [{min_L}, {max_L}], "
              f"neurons_window = [{cur_min_n}, {cur_max_n}], "
              f"learning_rates = {learning_rates}")

        # 1. Uruchamiamy GA w tym regionie
        best_sol, best_fit = genetic_algorithm(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_dim=input_dim,
            num_classes=num_classes,
            pop_size=base_pop_size,
            generations=base_generations,
            layers_range=(min_L, max_L),
            neurons_range=(cur_min_n, cur_max_n),
            learning_rates=learning_rates,
            mutation_rate=0.3,
            tournament_k=3,
            seed=42 + meta,   # lekko zmieniamy seed per meta-iterację
        )

        print(f"  -> region local best fitness: {best_fit:.4f}")

        # 2. Wyciągamy info o neuronach z najlepszego chromosomu
        best_neurons = extract_neurons_from_solution(best_sol)
        print(f"  -> best neurons in this run: {best_neurons}")

        # 3. Aktualizujemy region (fitness + zawężenie neuronów)
        qd.update_region(idx, new_best_fitness=best_fit, new_best_neurons=best_neurons)

        # 4. Globalne optimum
        if best_fit > global_best_fitness:
            global_best_fitness = best_fit
            global_best_solution = best_sol
            global_best_region = dict(region)

        best_idx, best_reg = qd.best_region_overall()
        print(f"  -> best region so far: {best_reg['name']} "
              f"(fitness={best_reg['best_fitness']:.4f}, "
              f"neurons_window={best_reg['current_neurons_range']})")
        print(f"  -> global best fitness so far: {global_best_fitness:.4f}")

    assert global_best_solution is not None
    assert global_best_region is not None

    return global_best_solution, global_best_fitness, global_best_region


# ============================================================
# 7. Example main
# ============================================================

def main():
    print("Ładuję dane MNIST (binarna klasyfikacja 0–4 vs 5–9)...")
    X_train, X_val, y_train, y_val, input_dim, num_classes = load_mnist_binary()
    print(f"  X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    print(f"  input_dim={input_dim}, num_classes={num_classes}")

    # Konfiguracja regionów dla QuantumDice
    regions_config = [
        {
            "name": "small_nets",
            "layers_range": (1, 2),
            "neurons_range": (16, 64),
            "learning_rates": [0.01, 0.001],
        },
        {
            "name": "medium_nets",
            "layers_range": (2, 3),
            "neurons_range": (32, 128),
            "learning_rates": [0.01, 0.001, 0.0005],
        },
        {
            "name": "deep_wide_nets",
            "layers_range": (3, 4),
            "neurons_range": (64, 256),
            "learning_rates": [0.001, 0.0005],
        },
    ]

    base_pop_size = 6       # dla demo; do poważnych eksperymentów zwiększ
    base_generations = 2    # j.w.
    meta_iters = 4          # ile razy rzucamy „QuantumDice”

    best_sol, best_fit, best_region = quantumdice_hyper_ga_adaptive_neurons(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=input_dim,
        num_classes=num_classes,
        base_pop_size=base_pop_size,
        base_generations=base_generations,
        regions_config=regions_config,
        meta_iters=meta_iters,
    )

    print("\n===== QuantumDice GA (adaptive neurons) – wynik końcowy =====")
    print("Najlepszy region:", best_region["name"])
    print("Parametry regionu:",
          "layers_range =", best_region["layers_range"],
          ", neurons_range =", best_region["neurons_range"],
          ", current_neurons_range =", best_region["current_neurons_range"],
          ", learning_rates =", best_region["learning_rates"])
    print("Najlepszy chromosom:",
          f"layers={best_sol.num_layers}, neurons={best_sol.num_neurons}, lr={best_sol.learning_rate}")
    print("Najlepsza uzyskana accuracy:", best_fit)


if __name__ == "__main__":
    main()
