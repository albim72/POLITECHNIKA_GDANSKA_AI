import numpy as np

# ============================================
# 1. Funkcja testowa: Rastrigin w d wymiarach
# ============================================

def rastrigin(x: np.ndarray) -> float:
    """
    Klasyczna funkcja Rastrigina.
    Minimum globalne: f(0,...,0) = 0.
    """
    A = 10.0
    x = np.asarray(x)
    return A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x))


# ============================================
# 2. QuantumDiceManager – meta-warstwa
#    zarządza centrami i promieniami
# ============================================

class QuantumDiceManager:
    """
    Prosty menedżer "kwantowej kostki":
    - utrzymuje listę centrów (regionów eksploracji),
    - każde centrum ma:
        center: wektor w R^d
        radius: skala eksploracji wokół center
        best_val: najlepsza znaleziona wartość w tym regionie
    - losuje centrum do eksploracji z rozkładu softmax(-beta * best_val)
      z domieszką rozkładu jednostajnego (eksploracja).
    """

    def __init__(
        self,
        n_dim: int,
        R0: float,
        max_centers: int = 5,
        beta: float = 1e-3,
        min_radius: float = 1e-2,
        max_radius: float | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.n_dim = n_dim
        self.R0 = R0
        self.min_radius = min_radius
        self.max_radius = max_radius if max_radius is not None else R0
        self.beta = beta
        self.rng = rng or np.random.default_rng()
        self.centers: list[dict] = []  # {center, radius, best_val}
        self.max_centers = max_centers

    def init_random_centers(self, k_init: int = 5):
        """
        Inicjalizacja kilku centrów losowo w szerokiej dziedzinie [-R0, R0]^d.
        """
        k_init = min(k_init, self.max_centers)
        for _ in range(k_init):
            center = self.rng.uniform(-self.R0, self.R0, size=self.n_dim)
            val = rastrigin(center)
            self.centers.append(
                {
                    "center": center,
                    "radius": self.R0,
                    "best_val": float(val),
                }
            )

    def get_probabilities(self) -> np.ndarray:
        """
        Rozkład prawdopodobieństwa nad centrami:
        im mniejsza best_val, tym większa waga.
        Używamy softmax(-beta * val) z normalizacją numeryczną.
        """
        vals = np.array([c["best_val"] for c in self.centers], dtype=float)
        m = np.min(vals)
        logits = -self.beta * (vals - m)
        exps = np.exp(logits - np.max(logits))
        probs = exps / exps.sum()
        return probs

    def choose_center_index(self) -> int:
        """
        Wybór indeksu centrum:
        mieszanka rozkładu rezonansowego (softmax) i jednostajnego (eksploracja).
        """
        if not self.centers:
            raise ValueError("Brak zainicjalizowanych centrów.")
        probs = self.get_probabilities()
        eps = 0.1  # domieszka eksploracji
        uniform = np.ones_like(probs) / len(probs)
        probs = (1.0 - eps) * probs + eps * uniform
        idx = self.rng.choice(len(self.centers), p=probs)
        return int(idx)

    def get_center(self, idx: int):
        c = self.centers[idx]
        return c["center"].copy(), float(c["radius"]), float(c["best_val"])

    def update_center(
        self,
        idx: int,
        new_center: np.ndarray,
        new_val: float,
        improve_shrink: float = 0.5,
        no_improve_expand: float = 1.2,
    ):
        """
        Aktualizacja centrum po lokalnym ES.

        - Jeśli jest poprawa: przesuwamy center w stronę new_center,
          aktualizujemy best_val, zmniejszamy radius (zoom).
        - Jeśli brak poprawy: lekko zwiększamy radius (szersza eksploracja).
        """
        c = self.centers[idx]
        if new_val < c["best_val"]:
            c["center"] = new_center.copy()
            c["best_val"] = float(new_val)
            c["radius"] = max(self.min_radius, c["radius"] * improve_shrink)
        else:
            c["radius"] = min(self.max_radius, c["radius"] * no_improve_expand)
        self.centers[idx] = c

    def best_global(self):
        """
        Zwraca globalnie najlepsze centrum.
        """
        vals = [c["best_val"] for c in self.centers]
        idx = int(np.argmin(vals))
        return self.centers[idx]["center"].copy(), float(self.centers[idx]["best_val"])


# ============================================
# 3. Lokalna strategia ewolucyjna w przestrzeni y ∈ [-1,1]^d
#    x = center + radius * y
# ============================================

def run_local_es(
    center: np.ndarray,
    radius: float,
    n_dim: int,
    lam: int = 200,
    mu_frac: float = 0.25,
    max_gen: int = 200,
    sigma_init: float = 0.3,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
):
    """
    Lokalny ES w znormalizowanej przestrzeni y ∈ [-1,1]^d, mapowany do
    x = center + radius * y.

    Zwraca:
    - best_x: najlepszy wektor w przestrzeni X,
    - best_val: jego wartość Rastrigina,
    - eval_count: ile razy wywołano funkcję celu.
    """
    rng = rng or np.random.default_rng(seed)
    mu = max(4, int(lam * mu_frac))

    # Populacja początkowa w y
    pop_y = rng.uniform(-1.0, 1.0, size=(mu, n_dim))
    pop_x = center + radius * pop_y
    scores = np.array([rastrigin(ind) for ind in pop_x])

    best_idx = int(np.argmin(scores))
    best_y = pop_y[best_idx].copy()
    best_x = pop_x[best_idx].copy()
    best_val = float(scores[best_idx])

    sigma = sigma_init
    eval_count = len(scores)

    for _gen in range(max_gen):
        # Wybór rodziców
        parents_idx = rng.integers(0, mu, size=lam)
        parents_y = pop_y[parents_idx]

        # Mutacje
        offspring_y = parents_y + sigma * rng.standard_normal(size=(lam, n_dim))
        offspring_y = np.clip(offspring_y, -1.0, 1.0)
        offspring_x = center + radius * offspring_y
        offspring_scores = np.array([rastrigin(ind) for ind in offspring_x])
        eval_count += len(offspring_scores)

        # Selekcja (μ + λ)
        cand_y = np.vstack([pop_y, offspring_y])
        cand_scores = np.concatenate([scores, offspring_scores])
        idx_sorted = np.argsort(cand_scores)
        pop_y = cand_y[idx_sorted[:mu]]
        scores = cand_scores[idx_sorted[:mu]]

        # Aktualizacja najlepszego w tym runie
        if scores[0] < best_val:
            best_val = float(scores[0])
            best_y = pop_y[0].copy()
            best_x = center + radius * best_y

        # Adaptacja sigma (reguła sukcesów)
        successes = np.sum(offspring_scores < best_val)
        success_rate = successes / lam
        if success_rate > 0.2:
            sigma *= 1.02
        else:
            sigma *= 0.98
        sigma = np.clip(sigma, 1e-3, 0.5)

    return best_x, best_val, eval_count


# ============================================
# 4. Główny algorytm:
#    QuantumDice + lokalne ES-y dla 100D
# ============================================

def quantumdice_es_rastrigin_100d(
    n_dim: int = 100,
    R0: float = 50.0,          # szeroka dziedzina: start w [-R0, R0]^d
    total_meta_iters: int = 20,
    lam: int = 200,
    mu_frac: float = 0.25,
    local_max_gen: int = 300,
    seed: int = 0,
):
    """
    Meta-algorytm:
    - inicjalizuje kilka centrów w [-R0, R0]^d,
    - wielokrotnie:
      * losuje centrum przez QuantumDiceManager,
      * uruchamia lokalny ES wokół tego centrum i radius,
      * aktualizuje centrum, promień i globalny optimum.

    Zwraca:
      - gbest_x: globalnie najlepszy wektor,
      - gbest_val: jego wartość,
      - total_evals: liczba wywołań funkcji celu.
    """
    rng = np.random.default_rng(seed)
    qd = QuantumDiceManager(
        n_dim=n_dim,
        R0=R0,
        max_centers=5,
        beta=1e-3,
        rng=rng,
    )
    qd.init_random_centers(k_init=5)

    total_evals = 0

    for meta in range(total_meta_iters):
        idx = qd.choose_center_index()
        center, radius, center_best_val = qd.get_center(idx)

        best_x, best_val, evals = run_local_es(
            center=center,
            radius=radius,
            n_dim=n_dim,
            lam=lam,
            mu_frac=mu_frac,
            max_gen=local_max_gen,
            sigma_init=0.3,
            rng=rng,
        )
        total_evals += evals

        qd.update_center(idx, best_x, best_val)
        gbest_x, gbest_val = qd.best_global()

        print(
            f"[Meta {meta+1:02d}] center={idx}, "
            f"local_best={best_val:.3f}, "
            f"center_prev_best={center_best_val:.3f}, "
            f"global_best={gbest_val:.3f}, "
            f"radius_used={radius:.3f}, "
            f"total_evals={total_evals}"
        )

    gbest_x, gbest_val = qd.best_global()
    return gbest_x, gbest_val, total_evals


# ============================================
# 5. Przykładowe uruchomienie
#    (do rzeczywistych eksperymentów podkręć parametry)
# ============================================

if __name__ == "__main__":
    # Lżejsze ustawienia do testu (możesz odpalić w 100D z większymi parametrami)
    best_x, best_val, total_evals = quantumdice_es_rastrigin_100d(
        n_dim=100,
        R0=50.0,
        total_meta_iters=20,    # zwiększ do 30–50, jeśli masz mocny CPU/GPU
        lam=200,                # możesz podnieść do 300–400
        mu_frac=0.25,
        local_max_gen=300,      # zwiększ do 400–600 dla lepszego zejścia
        seed=42,
    )

    print("\n=== WYNIK KOŃCOWY (QuantumDice ES, 100D) ===")
    print("Best value:", best_val)
    print("Norma x:", np.linalg.norm(best_x))
    print("Przykładowe pierwsze 5 współrzędnych x:", best_x[:5])
