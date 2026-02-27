# MCMC-PESP

## Cel projektu

Implementacja algorytmu MCMC (Markov Chain Monte Carlo) do rozwiązywania problemu PESP (Periodic Event Scheduling Problem) — optymalizacji rozkładów jazdy transportu publicznego. Projekt wykorzystuje miękkie ograniczenia, strukturalne propozycje ruchów i inercyjny Parallel Tempering.

Algorytm działa w dwóch trybach:
- **Tryb A (Sampling)**: formalny MCMC z próbkowaniem z rozkładu stacjonarnego
- **Tryb B (Optimization)**: metaheurystyka inspirowana MCMC (best-so-far)

## Quick Start

```bash
# Setup
cd mcmc-pesp
pip install -r requirements.txt

# Uruchom testy
pytest tests/

# PoC na syntetycznej sieci
python -m src.experiments.poc.triangle

# Benchmark na PESPlib
python -m src.experiments.benchmark.run --instance small
```

---

## Kluczowe założenia modelu (v2.0)

| Założenie | Wartość | Uwagi |
|-----------|---------|-------|
| Wspólny okres $T$ | 1440 min (1 doba) | Wszystkie linie: $T_L = T$ |
| Częstotliwość linii $f_L$ | Stała przez cały okres | $N_L = T / f_L$ kursów na okres |
| Przesiadki | Między **liniami**, nie kursami | $\gamma = (L_1, s_1, L_2, s_2, \rho)$ |
| Czas oczekiwania | $w_\gamma(x) = (-\Delta_\gamma(x)) \bmod f_{L_2}$ | Zawsze $\geq 0$, kawałkami liniowy |
| Ograniczenia | Miękkie (bariery kwadratowe) | Nigdy $+\infty$, gwarantuje ergodyczność |
| Robustność | Heurystyczne proxy oparte na CVaR opóźnień | Nie formalna minimalizacja CVaR straty |
| Proxy floty | Średnia intensywność (Little's Law) | $N^{\text{peak}}_{\text{conc}}$ jako diagnostyka, nie w koszcie |

---

## Struktura projektu

```
mcmc-pesp/
├── CLAUDE.md                 # Ten plik — instrukcje dla Claude Code
├── README.md                 # Dokumentacja użytkownika
├── requirements.txt          # Zależności Python
├── pyproject.toml           # Konfiguracja projektu
│
├── docs/
│   ├── MATH_SPEC.md         # Pełna specyfikacja matematyczna (v2.0)
│   ├── EXPERIMENTS.md       # Plan eksperymentów do przeprowadzenia
│   └── PESPLIB_FORMAT.md    # Dokumentacja formatu PESPlib
│
├── src/
│   ├── __init__.py
│   ├── core/                # Podstawowe struktury danych
│   │   ├── __init__.py
│   │   ├── network.py       # Station, Line, Network
│   │   ├── schedule.py      # Schedule (stan MCMC)
│   │   ├── events.py        # Event, Activity (EAN)
│   │   └── transfers.py     # Transfer, TransferParams
│   │
│   ├── cost/                # Funkcje kosztu
│   │   ├── __init__.py
│   │   ├── transfer_cost.py # Koszt przesiadek (soft constraints)
│   │   ├── operational.py   # Koszt operacyjny (turnaround, fleet proxy)
│   │   ├── robustness.py    # Proxy robustności (heurystyczne)
│   │   └── total.py         # Agregacja kosztów
│   │
│   ├── mcmc/                # Algorytmy MCMC
│   │   ├── __init__.py
│   │   ├── sampler.py       # Główna pętla MH
│   │   ├── proposals/       # Propozycje ruchów
│   │   │   ├── __init__.py
│   │   │   ├── base.py      # Interfejs Proposal
│   │   │   ├── single_line.py
│   │   │   ├── component.py
│   │   │   ├── smart_repair.py
│   │   │   └── swap.py
│   │   ├── parallel_tempering.py  # PT z inercją
│   │   └── adaptation.py    # Adaptacja wag i skal (Tryb A/B)
│   │
│   ├── metrics/             # Metryki i diagnostyka
│   │   ├── __init__.py
│   │   ├── quality.py       # Metryki jakości rozkładu
│   │   ├── convergence.py   # R-hat, ESS, trace plots
│   │   ├── peak_concurrency.py  # N_conc^peak (sweep-line)
│   │   └── comparison.py    # Porównanie z baseline
│   │
│   ├── io/                  # Wejście/wyjście
│   │   ├── __init__.py
│   │   ├── pesplib_loader.py    # Parser PESPlib → Network
│   │   ├── pesplib_exporter.py  # Schedule → PESPlib (walidacja)
│   │   ├── gtfs_loader.py       # Parser GTFS
│   │   └── synthetic.py         # Generatory syntetycznych sieci
│   │
│   └── experiments/         # Skrypty eksperymentów
│       ├── __init__.py
│       ├── poc/             # Proof of Concept
│       ├── benchmark/       # Benchmarki PESPlib
│       └── analysis/        # Analiza wyników
│
├── tests/                   # Testy jednostkowe i integracyjne
│   ├── __init__.py
│   ├── test_core/
│   ├── test_cost/
│   ├── test_mcmc/
│   └── test_io/
│
├── data/
│   ├── pesplib/
│   │   ├── raw/             # Oryginalne pliki PESPlib
│   │   └── converted/       # Skonwertowane do naszego formatu
│   ├── synthetic/           # Wygenerowane sieci testowe
│   └── gtfs/                # Dane GTFS (opcjonalnie)
│
├── configs/                 # Konfiguracje eksperymentów
│   ├── default.yaml
│   ├── poc.yaml
│   └── benchmark.yaml
│
└── notebooks/               # Jupyter notebooks do analizy
    ├── 01_poc_triangle.ipynb
    ├── 02_pesplib_validation.ipynb
    └── 03_benchmark_analysis.ipynb
```

---

## Kluczowe abstrakcje

### Core

```python
T_PERIOD = 1440  # Wspólny okres: minuty w dobie

@dataclass
class Station:
    id: str
    location: str      # Fizyczna lokalizacja
    modality: str      # Środek transportu

@dataclass
class Line:
    id: str
    modality: str
    stations: list[Station]
    travel_times: list[int]    # Minuty między stacjami
    dwell_times: list[int]     # Czasy postoju
    frequency: int             # Częstotliwość w minutach (np. 60)

    @property
    def n_trips(self) -> int:
        """Liczba kursów na okres T."""
        return T_PERIOD // self.frequency

@dataclass
class Schedule:
    """Stan MCMC — wektor offsetów linii"""
    offsets: dict[str, int]    # line_id → offset w minutach [0, T)
    
    def copy(self) -> Schedule: ...
    def shift_line(self, line_id: str, delta: int) -> None:
        """Przesuwa offset linii: (θ + δ) mod T"""
        ...

@dataclass
class Transfer:
    """Przesiadka MIĘDZY LINIAMI (nie kursami)"""
    id: str
    from_line: str             # L_1
    from_station: str          # s_1
    to_line: str               # L_2
    to_station: str            # s_2
    walk_time: int             # ρ — czas przejścia
    params: TransferParams     # w_min, w_ideal, α, β, M
    weight: float              # ω — waga (liczba pasażerów)

    def waiting_time(self, schedule: Schedule, network: Network) -> int:
        """w_γ(x) = (-Δ_γ(x)) mod f_{L_2}. Zawsze >= 0."""
        ...
```

### Cost

```python
class CostFunction(Protocol):
    def __call__(self, schedule: Schedule, network: Network) -> float: ...
    def delta(self, old: Schedule, new: Schedule, changed_lines: set[str]) -> float: ...

class TotalCost:
    """Cost(x) = Cost_PAX(x) + Cost_OPS(x) + Cost_ROB(x)"""
    def __init__(self, 
                 pax_weight: float = 1.0,
                 ops_weight: float = 1.0, 
                 rob_weight: float = 0.1): ...

class RobustnessCost:
    """Heurystyczne proxy robustności oparte na CVaR opóźnień.
    NIE jest formalną minimalizacją CVaR straty pasażera."""
    ...

class FleetProxyCost:
    """Średnia intensywność użycia taboru (Little's Law).
    NIE jest dolnym ograniczeniem na flotę.
    N_conc^peak raportowany osobno jako diagnostyka."""
    ...
```

### MCMC

```python
class Proposal(Protocol):
    """Interfejs dla propozycji ruchów"""
    def propose(self, schedule: Schedule, network: Network) -> tuple[Schedule, float]:
        """Zwraca (nowy_stan, hastings_ratio)"""
        ...
    
    @property
    def name(self) -> str: ...

class MCMCSampler:
    def __init__(self,
                 network: Network,
                 cost_fn: CostFunction,
                 proposals: list[Proposal],
                 proposal_weights: list[float],
                 beta: float,
                 mode: Literal["sampling", "optimization"] = "optimization"): ...
    
    def step(self) -> tuple[Schedule, bool]:
        """Jeden krok MCMC. Zwraca (stan, czy_zaakceptowano)"""
        ...
    
    def run(self, n_steps: int) -> MCMCResult: ...

class ProposalAdapter:
    """Adaptacja wag propozycji.
    Tryb A (sampling): diminishing γ_n = n^{-0.7}, p_j ≥ ε, σ ∈ [σ_min, σ_max]
    Tryb B (optimization): ciągła adaptacja bez ograniczeń"""
    ...

class ParallelTempering:
    def __init__(self,
                 network: Network,
                 cost_fn: CostFunction,
                 n_replicas: int,
                 beta_min: float,
                 beta_max: float,
                 inertia: dict[str, float] | None = None): ...
```

---

## Konwencje kodu

### Styl

- Python 3.11+
- Type hints **wszędzie** (strict mypy)
- Docstrings w formacie Google
- Max 100 znaków na linię
- `black` do formatowania
- `ruff` do lintingu

### Nazewnictwo

| Typ | Konwencja | Przykład |
|-----|-----------|----------|
| Klasy | PascalCase | `TransferCost`, `ComponentMove` |
| Funkcje/metody | snake_case | `compute_cost`, `propose_shift` |
| Stałe | UPPER_SNAKE | `T_PERIOD`, `DEFAULT_BETA` |
| Pliki | snake_case | `smart_repair.py`, `pesplib_loader.py` |

### Struktura pliku

```python
"""Krótki opis modułu."""

from __future__ import annotations

# Stdlib imports
import math
from dataclasses import dataclass
from typing import Protocol, Literal

# Third-party imports
import numpy as np

# Local imports
from src.core.network import Network

# Constants
T_PERIOD = 1440
DEFAULT_M_BARRIER = 1e6

# Classes/Functions
...
```

### Testy

- pytest jako framework
- Jeden plik testowy per moduł: `test_<module>.py`
- Nazwy testów: `test_<what>_<scenario>`
- Fixtures w `conftest.py`

```python
def test_waiting_time_always_non_negative():
    """w_γ(x) = (-Δ) mod f_{L2} jest zawsze >= 0."""
    ...

def test_waiting_time_modulo_frequency():
    """w_γ(x) ∈ [0, f_{L2})."""
    ...

def test_transfer_cost_below_minimum_returns_barrier():
    ...

def test_component_move_respects_decay_parameter():
    ...

def test_smart_repair_uses_minimal_norm():
    """δ* jest przesunięciem o minimalnej |δ|."""
    ...

def test_adaptation_weights_above_epsilon():
    """Tryb A: p_j(n) ≥ ε > 0 dla każdego j."""
    ...
```

---

## Workflow

### Dodawanie nowej propozycji ruchu

1. Stwórz plik `src/mcmc/proposals/<nazwa>.py`
2. Zaimplementuj interfejs `Proposal`:

```python
@dataclass
class MyNewMove:
    """Opis ruchu."""
    
    param1: float = 1.0
    
    @property
    def name(self) -> str:
        return "my_new_move"
    
    def propose(self, schedule: Schedule, network: Network) -> tuple[Schedule, float]:
        new_schedule = schedule.copy()
        # ... logika propozycji ...
        hastings_ratio = 1.0  # lub oblicz jeśli niesymetryczny
        return new_schedule, hastings_ratio
```

3. Dodaj testy w `tests/test_mcmc/test_my_new_move.py`
4. Zarejestruj w `src/mcmc/proposals/__init__.py`
5. Dodaj do konfiguracji w `configs/`

### Uruchamianie eksperymentów

```bash
# Pojedynczy eksperyment
python -m src.experiments.poc.triangle --config configs/poc.yaml

# Benchmark na PESPlib
python -m src.experiments.benchmark.run \
    --instances small medium \
    --methods mcmc sa random \
    --output results/benchmark_001/

# Analiza zbieżności
python -m src.experiments.analysis.convergence \
    --input results/benchmark_001/ \
    --output results/analysis_001/
```

---

## Priorytety implementacji

### Faza 1: Core + PoC (tydzień 1-2) ✅ DONE

- [x] `src/core/` — struktury danych
- [x] `src/cost/transfer_cost.py` — koszt z barierą
- [x] `src/mcmc/proposals/single_line.py` — Single Line Shift
- [x] `src/mcmc/sampler.py` — pętla MH
- [x] `src/io/synthetic.py` — generator sieci Triangle
- [x] `tests/`

### Faza 2: Propozycje + Metryki (tydzień 3-4) ✅ DONE

- [x] `src/mcmc/proposals/component.py` — Component Move
- [x] `src/mcmc/proposals/smart_repair.py` — Smart Repair
- [x] `src/cost/total.py` — pełna funkcja kosztu
- [x] `src/metrics/convergence.py` — R-hat, ESS

### Faza 2.5: Poprawki modelu v2.0 ✅ DONE (2026-02-13)

- [x] **Refaktor przesiadek**: `Transfer` między liniami, nie kursami
- [x] **Czas oczekiwania**: $w_\gamma(x) = (-\Delta) \bmod f_{L_2}$, zawsze $\geq 0$
- [x] **Wspólny okres**: $T = 1440$, `Line.frequency` zamiast `Line.period`
- [x] **Ergodyczność**: upewnić się że kernel daje $P(L) > 0$ i $\delta$ ma wsparcie $\mathbb{Z}$
- [x] **Robustness**: rename/docstring — „proxy", nie „CVaR minimization"
- [x] **Smart Repair**: reguła minimalnej normy dla $\delta^*$
- [x] **Adaptacja**: dodać `mode` (sampling/optimization), containment ($p_j \geq \varepsilon$)
- [x] **Fleet proxy**: docstring — „średnia intensywność", dodać `peak_concurrency.py`
- [x] **Testy**: nowe testy dla $w \geq 0$, minimalnej normy, containment
- [x] **Discrete Hastings**: SmartRepair używa dyskretnych mas prawdopodobieństwa
- [x] **ProposalAdapter integration**: MCMCSampler integruje adapter w Trybie A

**Kryterium sukcesu**: ✅ Wszystkie testy przechodzą (271), model matematycznie spójny z MATH_SPEC v2.0.

### Faza 2.6: Mode A/B Formal Compliance ✅ DONE (2026-02-13)

- [x] **SmartRepair discrete Hastings**: Używa `_compute_discrete_mass()` dla detailed balance
- [x] **ProposalAdapter integration**: Sampler tworzy adapter w mode="sampling"
- [x] **Diminishing adaptation**: Gamma_n = n^{-0.7} → 0 w Trybie A
- [x] **Containment**: p_j ≥ ε = 0.05 dla wszystkich propozycji
- [x] **Mode B unchanged**: Optimization mode działa jak wcześniej (best-so-far)
- [x] **Testy**: 8 nowych testów dla discrete mass i adapter integration

**Rezultat**: Mode A (sampling) jest teraz formalnie poprawny. Mode B (optimization) bez zmian.

### Faza 3: Parallel Tempering (tydzień 5-6) ✅ DONE (2026-02-14)

- [x] `src/mcmc/parallel_tempering.py` — PT z inercją (InertiaConfig, PTResult, ParallelTempering)
- [x] `src/mcmc/proposals/single_line.py` — `line_weights` dla inercji
- [x] `src/mcmc/proposals/component.py` — `line_weights` dla inercji
- [x] `tests/test_mcmc/test_parallel_tempering.py` — 34 testy
- [x] `src/experiments/poc/triangle_pt.py` — PoC porównanie PT vs single chain
- [x] `src/cost/robustness.py` — proxy robustności (already done in Faza 2)
- [x] `src/cost/operational.py` — turnaround, fleet proxy (already done in Faza 2)

**Rezultat**: 305 testów przechodzi. PT z inercją działa na syntetycznej sieci.
Swap acceptance rates w zakresie 20-94% (zdrowe). Benchmark vs SA, random search — do Fazy 4.

### Faza 4: Skalowanie + GTFS (tydzień 7+)

- [ ] `src/io/gtfs_loader.py`
- [ ] Optymalizacje wydajności (inkrementalny koszt)
- [ ] Duże instancje PESPlib
- [ ] Rzeczywiste dane

---

## Ważne zasady

### DO

- ✅ Zawsze pisz testy przed/wraz z implementacją
- ✅ Używaj type hints i sprawdzaj mypy
- ✅ Loguj ważne metryki (acceptance rate, koszt, czas)
- ✅ Dokumentuj założenia i decyzje w docstringach
- ✅ Używaj `dataclass` lub `NamedTuple` dla struktur danych
- ✅ Profiluj przed optymalizacją
- ✅ Rozróżniaj Tryb A (sampling) od Tryb B (optimization) w kodzie

### DON'T

- ❌ Nie hardcoduj parametrów — używaj configów
- ❌ Nie implementuj wszystkiego naraz — iteruj
- ❌ Nie ignoruj warningów mypy/ruff
- ❌ Nie optymalizuj przedwcześnie
- ❌ Nie commituj bez działających testów
- ❌ Nie używaj `print()` — używaj `logging`
- ❌ Nie nazywaj proxy robustności „CVaR minimization" — to heurystyczne proxy

### Znane pułapki

1. **Arytmetyka modularna**: Offsety są mod T = 1440. Używaj `(theta + delta) % T_PERIOD`, nie `theta + delta`.

2. **Czas oczekiwania**: `w = (-delta_phase) % frequency`. Wynik jest ZAWSZE w `[0, frequency)`. Nigdy nie powinien być ujemny — jeśli jest, to bug w obliczeniu `delta_phase`.

3. **Przesiadki między liniami, nie kursami**: Transfer definiuje relację $(L_1, s_1, L_2, s_2, \rho)$. Przy stałej częstotliwości czas oczekiwania jest identyczny dla każdego kursu — nie iteruj po kursach.

4. **Hastings ratio**: Dla niesymetrycznych propozycji (Smart Repair) pamiętaj o `q(x|x') / q(x'|x)`. Smart Repair używa reguły minimalnej normy dla $\delta^*$ i **dyskretnych mas prawdopodobieństwa** (nie ciągłych gęstości) dla detailed balance: `q_k = Φ((k+0.5-δ*)/σ) - Φ((k-0.5-δ*)/σ)` gdzie k jest całkowitym przesunięciem.

5. **Inkrementalny koszt**: `delta_cost()` musi zwracać **dokładnie** tę samą wartość co `cost(new) - cost(old)`. Testuj to!

6. **Parallel Tempering**: Repliki muszą być niezależne. Nie dziel mutowalnych obiektów.

7. **Adaptacja wag (Tryb A)**: Wagi $p_j$ muszą spełniać $p_j \geq \varepsilon > 0$. Skala $\sigma \in [\sigma_{\min}, \sigma_{\max}]$. Krok adaptacji $\gamma_n = n^{-0.7} \to 0$.

8. **Proxy floty**: `N_fleet` to średnia intensywność (Little's Law), NIE peak concurrency. Nie traktuj jako twardego ograniczenia na flotę.

9. **Nieciągłość $w_\gamma(x)$**: Czas oczekiwania ma skok z 0 do $f_{L_2}$ — to normalne (zmiana „łapanego" kursu). MCMC nie wymaga ciągłości kosztu, tylko obliczalności.

---

## Referencje

- **Specyfikacja matematyczna**: `docs/MATH_SPEC.md` (v2.0)
- **Plan eksperymentów**: `docs/EXPERIMENTS.md`
- **PESPlib**: https://num.math.uni-goettingen.de/~m.goerigk/pesplib/
- **Format PESPlib**: `docs/PESPLIB_FORMAT.md`
