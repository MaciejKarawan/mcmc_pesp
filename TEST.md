# Plan: Implementacja Frameworku Syntetycznych Benchmarków MCMC-PESP

## Context

Projekt MCMC-PESP jest obecnie w późnej Fazie 2 (PESPlib + Propozycje). Wszystkie komponenty są zaimplementowane:
- Core: Network, Schedule, Transfer (623 linie)
- Cost: TransferCost, OperationalCost, RobustnessCost, TotalCost (1,250 linii)
- MCMC: Sampler + propozycje (SingleLineShift, ComponentMove, SmartRepair)
- Testy: 4,721 linii z dobrą pokryciem

**Problem**: Dotychczasowe testowanie opierało się na PESPlib, który nie pasuje do modelu miękich ograniczeń. Obecne syntetyczne sieci (Triangle: 1 transfer, Mini: 2 transfery) są zbyt proste do rzeczywistego stress testingu MCMC.

**Cel**: Stworzenie frameworku syntetycznych benchmarków pokrywającego:
1. **Konflikty między transferami** - multimodalne cost landscape
2. **Skalowanie** - sieci 5-10 linii (hub, łańcuch)
3. **Heterogeniczne wagi pasażerów** - testowanie priorytetyzacji
4. **Multi-komponentowy koszt** - transfer + operational + robustness

**Korzyści**:
- Systematyczne testowanie MCMC pod różnymi warunkami
- Znane optima dla walidacji (hub, chain)
- Parametryzowalne scenariusze (difficulty: easy/medium/hard)
- Benchmark proposal effectiveness per scenario

---

## Podejście: Factory Pattern + 5 Scenariuszy

### Architektura

```
src/io/synthetic_scenarios.py         [NOWY, ~700 linii]
├── ScenarioMetadata (dataclass)
├── SyntheticScenario (dataclass)
└── SyntheticScenarioFactory
    ├── create(scenario_type, difficulty, **params)
    └── Scenario builders:
        ├── _build_conflicting_transfers()
        ├── _build_hub_network()
        ├── _build_chain_network()
        ├── _build_weighted_transfers()
        └── _build_full_cost_scenario()

src/experiments/benchmark/run_synthetic_benchmark.py  [NOWY, ~500 linii]
├── SyntheticBenchmarkConfig (dataclass)
├── ScenarioBenchmarkResult (dataclass)
├── run_scenario_benchmark()
└── run_synthetic_benchmark_suite()

configs/synthetic_benchmark.yaml      [NOWY, ~80 linii]
tests/test_io/test_synthetic_scenarios.py          [NOWY, ~300 linii]
tests/test_experiments/test_synthetic_benchmark.py [NOWY, ~200 linii]
```

### 5 Scenariuszy Testowych

#### 1. **ConflictingTransfersNetwork**
- **Struktura**: 3 linie spotykające się w hubie H
- **Transfery**: T1→2 (w_ideal=8), T2→3 (w_ideal=12), T3→1 (w_ideal=15)
- **Konflikt**: Optymalizacja jednego transferu pogarsza pozostałe
- **Difficulty**: easy=2 konflikty, medium=3, hard=4+
- **Optimum**: Brak (NP-hard), empiryczny cost range: 50-200

#### 2. **HubNetwork**
- **Struktura**: Hub centralny z N promieniowymi liniami
- **Transfery**: Wszystkie przez hub (inbound→outbound)
- **Difficulty**: easy=5 linii/6 transferów, medium=8/12, hard=12/24
- **Optimum**: **ZNANE** - topologia gwiazdy ma closed-form solution
- **Cel**: Testowanie skalowania

#### 3. **ChainNetwork**
- **Struktura**: Sekwencyjny łańcuch L1→L2→L3→...→LN
- **Transfery**: N-1 transferów w kolejnych węzłach
- **Difficulty**: easy=3 linie, medium=5, hard=8
- **Optimum**: **ZNANE** - propagacja offsetów: θᵢ₊₁ = (θᵢ + 17) mod T
- **Cel**: Testowanie long-range dependencies

#### 4. **WeightedTransfersNetwork**
- **Struktura**: Hub network z heterogenicznymi wagami
- **Wagi**: Niektóre transfery weight=1, inne weight=100
- **Difficulty**: easy=2 high/4 low, medium=4/8, hard=6/18
- **Cel**: Testowanie priorytetyzacji przez MCMC

#### 5. **FullCostNetwork**
- **Struktura**: Hub z bidirectional lines (out+return)
- **Koszty**: Transfer + Turnaround + Robustness (CVaR)
- **Difficulty**: easy=2 linie/2 turnarounds, medium=4/6, hard=6/12
- **Cel**: Multi-objective optimization

### Kluczowe Struktury Danych

```python
@dataclass
class ScenarioMetadata:
    name: str
    description: str
    n_lines: int
    n_transfers: int
    difficulty: str  # "easy" | "medium" | "hard"
    expected_cost_range: tuple[float, float] | None
    optimal_solution: dict[str, int] | None  # Known optimum offsets
    properties: dict[str, Any]  # {"conflicting": True, "hub": True}

@dataclass
class SyntheticScenario:
    network: Network
    transfers: TransferSet
    metadata: ScenarioMetadata
    turnarounds: TurnaroundSet | None = None

    def compute_cost(self, schedule: Schedule, cost_type: str = "transfer_only") -> float:
        """Compute cost for a schedule under this scenario."""

@dataclass
class ScenarioBenchmarkResult:
    scenario_name: str
    metadata: ScenarioMetadata
    best_cost: float
    best_solution: dict[str, int]
    optimal_cost: float | None
    gap_to_optimum: float | None  # (best - optimal) / optimal
    acceptance_rate: float
    diagnostics: dict[str, Any]  # R-hat, ESS, converged
    proposal_effectiveness: dict[str, float]
    runtime_seconds: float
```

---

## Pliki do Stworzenia/Modyfikacji

### Pliki Główne

**1. `src/io/synthetic_scenarios.py`** [NOWY, ~700 linii]
- Implementacja factory pattern
- 5 scenario builders
- Reużycie wzorców z `src/io/synthetic.py` (TriangleNetwork)
- Type hints, docstrings w formacie Google

**2. `src/experiments/benchmark/run_synthetic_benchmark.py`** [NOWY, ~500 linii]
- Orchestration benchmarków
- Reużycie infrastruktury multi-chain MCMC z `run_pesplib.py`
- Diagnostyka: R-hat, ESS, convergence detection
- Output: JSON (pełne wyniki) + CSV (summary table)
- CLI entry point z argparse

**3. `configs/synthetic_benchmark.yaml`** [NOWY, ~80 linii]
- Lista scenariuszy do uruchomienia
- Parametry MCMC (n_chains=4, n_steps=20k, beta=1.0)
- Wagi propozycji (single_line=0.4, component=0.3, smart_repair=0.2, uniform=0.1)
- Ustawienia output

### Pliki Testowe

**4. `tests/test_io/test_synthetic_scenarios.py`** [NOWY, ~300 linii]
- `TestSyntheticScenarioFactory`: tworzenie każdego scenariusza
- `TestScenarioStructure`: walidacja n_lines, n_transfers, conflicts
- `test_known_optimum_gives_zero_cost()`: hub i chain optima

**5. `tests/test_experiments/test_synthetic_benchmark.py`** [NOWY, ~200 linii]
- `test_run_scenario_benchmark_hub_easy()`: pojedynczy benchmark
- `test_benchmark_finds_known_optimum()`: gap < 5% dla chain
- `test_full_benchmark_suite()`: integracja end-to-end

### Pliki Referencyjne (bez zmian)

**6. `src/io/synthetic.py`** [REFERENCJA]
- Wzorzec TriangleNetwork do naśladowania
- Pozostaje bez zmian, opcjonalnie wrapper w scenarios

**7. `src/experiments/benchmark/run_pesplib.py`** [REFERENCJA]
- Wzorzec multi-chain benchmarku do reużycia
- Struktura MCMCResult, diagnostics

---

## Kolejność Implementacji

### Faza 1: Core Infrastructure (2-3 dni)
**Cel**: Działający factory + 1 scenariusz

1. **Stwórz `src/io/synthetic_scenarios.py`**
   - Dataclasses: `ScenarioMetadata`, `SyntheticScenario`
   - Factory skeleton: `SyntheticScenarioFactory` z `create()` i `list_scenarios()`

2. **Implementuj HubNetwork** (najprostszy, znane optimum)
   - `_build_hub_network(difficulty, period, seed, **kwargs)`
   - Difficulty levels: easy=5 linii, medium=8, hard=12
   - Compute analytical optimum: inbound offset=0, outbound offset=17+stagger

3. **Napisz testy dla hub**
   - `test_create_hub_network_easy()`
   - `test_hub_network_has_correct_structure()`
   - `test_known_optimum_gives_low_cost()`

4. **Walidacja**: Factory tworzy poprawny Network + TransferSet

---

### Faza 2: Dodatkowe Scenariusze (3-4 dni)
**Cel**: Wszystkie 5 scenariuszy działające

5. **Implementuj ChainNetwork**
   - Sequential L1→L2→...→LN
   - Analytical optimum: θᵢ₊₁ = (θᵢ + travel_time + dwell + w_ideal) % period

6. **Implementuj ConflictingTransfersNetwork**
   - 3 linie w triangle z konfliktowymi w_ideal
   - Brak analytical optimum, empiryczny cost range

7. **Implementuj WeightedTransfersNetwork**
   - Reużyj hub topology
   - Randomizuj wagi: niektóre 1.0, niektóre weight_ratio (default=100.0)

8. **Implementuj FullCostNetwork**
   - Hub + bidirectional lines
   - Include TurnaroundSet
   - `compute_cost()` używa TotalCost zamiast TransferCost

9. **Napisz testy dla każdego scenariusza**
   - Structure validation
   - Cost computation
   - Known optima gdzie możliwe

10. **Integration test**: Stwórz wszystkie scenariusze, verify properties

---

### Faza 3: Benchmark Runner (2-3 dni)
**Cel**: Automatyczne uruchamianie benchmarków

11. **Stwórz `src/experiments/benchmark/run_synthetic_benchmark.py`**
    - Dataclasses: `SyntheticBenchmarkConfig`, `ScenarioBenchmarkResult`, `SyntheticBenchmarkSuite`

12. **Implementuj `run_scenario_benchmark()`**
    - Przyjmuje `SyntheticScenario` + config
    - Tworzy odpowiednią cost function (TransferCost vs TotalCost)
    - Uruchamia multi-chain MCMC (reużyj pattern z `run_pesplib.py`)
    - Compute diagnostics: R-hat, ESS, convergence
    - Compute gap to optimum jeśli znane
    - Aggregate proposal statistics

13. **Implementuj `run_synthetic_benchmark_suite()`**
    - Iteruje po scenariuszach z config
    - Używa factory do tworzenia
    - Zbiera wyniki
    - Compute summary statistics

14. **CLI entry point**
    - argparse: `--config`, `--output`, `--seed`
    - Load YAML config
    - Run suite
    - Save JSON + CSV

15. **Integration tests**
    - `test_run_scenario_benchmark_hub_easy()`: 2 chains × 500 steps
    - `test_benchmark_finds_known_optimum()`: chain scenario, gap < 5%

---

### Faza 4: Configuration & Validation (1-2 dni)
**Cel**: Końcowa walidacja + dokumentacja

16. **Stwórz `configs/synthetic_benchmark.yaml`**
    - Define suite: 9 scenariuszy (conflicting easy/medium, hub easy/medium/hard, chain easy/medium, weighted medium, full_cost easy)
    - MCMC params: 4 chains, 20k steps, burn_in=0.1
    - Proposal weights: single_line=0.4, component=0.3, smart_repair=0.2, uniform=0.1

17. **Test loading config**
    - `load_benchmark_config()` z YAML
    - Validate schema

18. **Run mini benchmark**
    - 2 scenariusze (hub easy, chain easy)
    - 2 chains × 1000 steps
    - Verify output JSON/CSV format

19. **Validate known optima**
    - Hub easy: gap < 10% (może nie być perfect ze względu na stochastyczność)
    - Chain easy: gap < 5%

---

### Faza 5: Dokumentacja & Polish (1 dzień)
**Cel**: Produkcyjny kod + instrukcje użycia

20. **Docstrings**
    - Google format dla wszystkich funkcji
    - Type hints wszędzie (mypy strict)

21. **Aktualizuj `CLAUDE.md`**
    - Sekcja "Synthetic Benchmarks" z przykładami użycia
    - Update workflow

22. **Opcjonalnie: Notebook**
    - `notebooks/04_synthetic_benchmark.ipynb`
    - Load results, plot cost vs difficulty
    - Analyze proposal effectiveness

23. **Run full benchmark suite**
    - Wszystkie 9 scenariuszy
    - 4 chains × 20k steps
    - Dokumentuj baseline results w README

---

## Wzorce Implementacji

### Reużycie Existing Patterns

**Z `src/io/synthetic.py` (TriangleNetwork)**:
- Dataclass pattern z konfigurowalnymi parametrami
- `compute_optimal_offset()` method dla known optima
- Helper methods do tworzenia Station/Line/Transfer

**Z `src/experiments/benchmark/run_pesplib.py`**:
- Multi-chain MCMC loop
- MCMCResult dataclass
- Diagnostics computation (R-hat, ESS)
- Result serialization

**Z `tests/conftest.py`**:
- Fixtures dla Network, TransferSet
- Parametryzacja testów (@pytest.mark.parametrize)

### Przykładowy Kod: Hub Network Builder

```python
def _build_hub_network(
    difficulty: str,
    period: int = 60,
    seed: int | None = None,
    n_lines: int | None = None,
    **kwargs
) -> SyntheticScenario:
    """Build hub-and-spoke network.

    Args:
        difficulty: "easy" (5 lines), "medium" (8), "hard" (12)
        period: Schedule period in minutes
        seed: Random seed (unused for hub, deterministic)
        n_lines: Override default n_lines for difficulty
        **kwargs: Additional parameters (ignored)

    Returns:
        SyntheticScenario with hub topology and known optimum.
    """
    if n_lines is None:
        n_lines = {"easy": 5, "medium": 8, "hard": 12}[difficulty]

    # Create hub station
    hub = Station(id="Hub", location="Hub", modality="bus")

    # Create radial lines (alternating inbound/outbound)
    lines = []
    for i in range(n_lines):
        endpoint = Station(id=f"End{i}", location=f"End{i}", modality="bus")
        direction = "inbound" if i % 2 == 0 else "outbound"

        if direction == "inbound":
            stations = [endpoint, hub]
        else:
            stations = [hub, endpoint]

        line = Line(
            id=f"L{i+1}",
            modality="bus",
            stations=stations,
            travel_times=[15],
            dwell_times=[2, 2],
            period=period
        )
        lines.append(line)

    network = Network(lines=lines, period=period)

    # Create transfers: inbound → outbound at hub
    transfers = {}
    transfer_params = TransferParams(
        w_min=3.0, w_ideal=8.0, w_max=30.0, alpha=2.0, beta=0.5
    )

    for i in range(0, n_lines, 2):  # Inbound
        for j in range(1, n_lines, 2):  # Outbound
            # Transfer from Li arrival to Lj departure
            transfer = Transfer(
                id=f"T_L{i+1}_L{j+1}",
                from_line=f"L{i+1}",
                from_station=hub.location,
                to_line=f"L{j+1}",
                to_station=hub.location,
                walk_time=3.0,
                params=transfer_params,
                weight=1.0
            )
            transfers[transfer.id] = transfer

    # Compute analytical optimum
    # Inbound lines: offset = 0
    # Outbound lines: offset = travel_time + dwell + w_ideal = 15 + 2 + 8 = 25
    # Stagger outbound slightly to avoid simultaneous departures
    optimal_offsets = {}
    for i in range(n_lines):
        if i % 2 == 0:  # Inbound
            optimal_offsets[f"L{i+1}"] = 0
        else:  # Outbound
            optimal_offsets[f"L{i+1}"] = 25 + (i // 2) * 1  # Stagger by 1 min

    metadata = ScenarioMetadata(
        name=f"hub_{n_lines}_lines_{difficulty}",
        description=f"Hub-and-spoke with {n_lines} radial lines",
        n_lines=n_lines,
        n_transfers=len(transfers),
        difficulty=difficulty,
        expected_cost_range=(0.0, 5.0),
        optimal_solution=optimal_offsets,
        properties={"hub": True, "scalability": True}
    )

    return SyntheticScenario(
        network=network,
        transfers=TransferSet(transfers=transfers),
        metadata=metadata
    )
```

---

## Weryfikacja End-to-End

### Kryteria Sukcesu

Po implementacji, następujące testy muszą przejść:

1. **Factory Creation**
   ```python
   factory = SyntheticScenarioFactory()
   scenario = factory.create("hub", difficulty="easy")
   assert scenario.metadata.n_lines == 5
   assert scenario.metadata.optimal_solution is not None
   ```

2. **Known Optimum (Hub)**
   ```python
   scenario = factory.create("hub", difficulty="easy")
   optimal_schedule = Schedule(
       offsets=scenario.metadata.optimal_solution,
       period=60
   )
   cost = TransferCost(transfers=scenario.transfers)(optimal_schedule, scenario.network)
   assert cost < 1.0  # Near-zero
   ```

3. **MCMC Convergence (Chain)**
   ```python
   config = SyntheticBenchmarkConfig(
       scenarios=[{"type": "chain", "difficulty": "easy"}],
       n_chains=4,
       n_steps_per_chain=10000
   )
   suite = run_synthetic_benchmark_suite(config, seed=42)
   result = suite.results[0]
   assert result.gap_to_optimum < 0.05  # Within 5%
   assert result.diagnostics["converged"] is True
   ```

4. **Full Benchmark Suite**
   ```bash
   python -m src.experiments.benchmark.run_synthetic_benchmark \
       --config configs/synthetic_benchmark.yaml \
       --output results/synthetic_001 \
       --seed 42

   # Check output files exist
   ls results/synthetic_001/results.json
   ls results/synthetic_001/summary.csv
   ```

5. **Proposal Effectiveness**
   - ComponentMove powinien mieć wyższą acceptance rate na hub/chain
   - SmartRepair powinien być użyteczny na conflicting transfers
   - Verify w summary.csv kolumna `proposal_effectiveness`

### Expected Outputs

**`results/synthetic_001/summary.csv`**:
```csv
scenario_name,difficulty,n_lines,n_transfers,best_cost,optimal_cost,gap,acceptance_rate,r_hat,converged,runtime_sec
hub_5_lines_easy,easy,5,6,1.2,0.0,NA,0.42,1.03,true,12.5
hub_8_lines_medium,medium,8,12,3.8,0.0,NA,0.38,1.05,true,25.3
chain_3_lines_easy,easy,3,2,0.05,0.0,0.05,0.40,1.02,true,8.2
conflicting_transfers_medium,medium,3,3,87.4,NA,NA,0.35,1.08,true,15.3
weighted_medium,medium,8,12,245.7,NA,NA,0.36,1.06,true,28.1
```

**`results/synthetic_001/results.json`** (excerpt):
```json
{
  "config": { ... },
  "results": [
    {
      "scenario_name": "hub_5_lines_easy",
      "metadata": {
        "name": "hub_5_lines_easy",
        "n_lines": 5,
        "optimal_solution": {"L1": 0, "L2": 25, "L3": 0, "L4": 26, "L5": 0}
      },
      "best_cost": 1.2,
      "best_solution": {"L1": 0, "L2": 24, "L3": 1, "L4": 26, "L5": 0},
      "gap_to_optimum": null,
      "acceptance_rate": 0.42,
      "diagnostics": {
        "r_hat": 1.03,
        "ess_bulk": 4500,
        "converged": true
      },
      "proposal_effectiveness": {
        "single_line": 0.38,
        "component": 0.48,
        "smart_repair": 0.35
      }
    },
    ...
  ],
  "summary": {
    "total_scenarios": 9,
    "converged": 8,
    "mean_acceptance_rate": 0.38
  }
}
```

---

## Potencjalne Pułapki

1. **Transfer station indexing**: Upewnij się że `from_station_idx` i `to_station_idx` są prawidłowe dla topologii
2. **Period wraparound**: Optimal offsets muszą być mod period
3. **Hub topology**: Inbound vs outbound direction - station order matters
4. **Known optimum precision**: Ze względu na stochastyczność, cost może nie być dokładnie 0.0
5. **Conflicting scenario**: Verify że faktycznie występują konflikty (optymalizacja jednego pogarsza drugi)
6. **TotalCost w full_cost**: Upewnij się że TurnaroundSet jest poprawnie podłączony

---

## Migracja z Istniejącego Kodu

### Zachowanie TriangleNetwork

Existing `src/io/synthetic.py` **bez zmian**. Opcjonalnie wrapper:

```python
def _build_triangle_wrapper(difficulty: str, period: int, **kwargs) -> SyntheticScenario:
    """Wrapper around existing TriangleNetwork for backward compatibility."""
    builder = TriangleNetwork(period=period)
    # ... convert to SyntheticScenario ...
```

### Update PoC Scripts (Opcjonalnie)

Existing `src/experiments/poc/triangle.py` działa bez zmian. Nowy skrypt:

```python
# src/experiments/poc/scenario_poc.py
from src.io.synthetic_scenarios import SyntheticScenarioFactory

factory = SyntheticScenarioFactory()
scenario = factory.create("hub", difficulty="easy")
# ... run MCMC as before ...
```

---

## Podsumowanie

**Dostarczone:**
- Factory pattern dla 5 typów scenariuszy testowych
- Hub i Chain z known optima (walidacja MCMC)
- Conflicting, Weighted, FullCost dla stress testing
- Benchmark runner z multi-chain MCMC + diagnostyką
- Comprehensive tests (unit + integration + regression)
- YAML config dla parametryzacji

**Szacowany nakład**: 10-15 dni roboczych dla pełnej, przetestowanej, udokumentowanej implementacji.

**Kluczowe pliki do implementacji**:
1. `src/io/synthetic_scenarios.py` (~700 linii)
2. `src/experiments/benchmark/run_synthetic_benchmark.py` (~500 linii)
3. `configs/synthetic_benchmark.yaml` (~80 linii)
4. `tests/test_io/test_synthetic_scenarios.py` (~300 linii)
5. `tests/test_experiments/test_synthetic_benchmark.py` (~200 linii)

**Łącznie**: ~1,780 linii nowego kodu + dokumentacja.
