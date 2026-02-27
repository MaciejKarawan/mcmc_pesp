# Instrukcja obslugi MCMC-PESP

Ten dokument opisuje **jak uzywac kodu**, jakie sa parametry, skad biora sie defaulty i jak uruchamiac testy (w tym testy na instancjach).
Formalna matematyka pozostaje w `docs/MATH_SPEC.md`.

## 1. Co uruchamiasz i po co

W repo sa 3 glowne sciezki uruchomieniowe:

1. `python -m src.experiments.poc.triangle`
   - szybki Proof of Concept na malej sieci.
2. `python -m src.experiments.benchmark.run_synthetic_benchmark`
   - benchmarki na scenariuszach syntetycznych (`hub`, `chain`, `conflicting`, `full_cost`, itd.).
3. `python -m src.experiments.benchmark.run_pesplib` oraz `python -m src.experiments.benchmark.run_pesplib_raw`
   - benchmarki na danych w formacie PESPlib.

## 2. Tryby algorytmu

Kod sampler-a wspiera:

- `mode="optimization"` (domyslnie w `MCMCSampler`): sledzenie best-so-far, podejscie metaheurystyczne.
- `mode="sampling"`: formalny MCMC z adaptacja malejaca (diminishing adaptation) przez `ProposalAdapter`.

W praktyce skrypty benchmarkowe w tym repo domyslnie uruchamiaja tryb optymalizacyjny, chyba ze jawnie zmienisz tryb przy tworzeniu `MCMCSampler`.

## 3. Jak dziala MH w implementacji

W `src/mcmc/sampler.py` krok ma postac:

- propozycja: `new_schedule, hastings_ratio, changed_lines = proposal.propose(...)`
- inkrement kosztu: `delta_cost = cost_fn.delta(old, new, changed_lines, network)`
- akceptacja:
  - `log_alpha = -beta * delta_cost + log(hastings_ratio)`
  - akceptuj jesli `log_alpha >= 0` lub z prawdopodobienstwem `exp(log_alpha)`.

To odpowiada MH: `min(1, exp(-beta*DeltaCost) * q(x|y)/q(y|x))`.

## 4. Parametry i defaulty: warstwy nadpisywania

Praktyczna kolejnosc priorytetu (od najwyzszego):

1. CLI (`--...`) w skrypcie.
2. YAML (jesli skrypt czyta konfiguracje z pliku).
3. Defaulty w dataclass/funkcjach budujacych obiekty.

Przyklad (`run_synthetic_benchmark`):

- CLI: `--config`, `--output`, `--seed`
- YAML: sekcja `mcmc:` i `scenarios:` w `configs/synthetic_benchmark.yaml`
- fallback: `SyntheticBenchmarkConfig` w `src/experiments/benchmark/run_synthetic_benchmark.py`

## 5. Defaulty kluczowych parametrow (algorytm)

### 5.1 Sampler i chain

`src/mcmc/sampler.py`:

- `beta=1.0`
- `mode="optimization"`
- `proposal_weights=None` -> rowne wagi, jesli nie podasz.
- `run(...): thin=1, burn_in=0, log_every=1000`

Dlaczego:
- `beta=1.0` to neutralny punkt startowy (ani bardzo greed, ani bardzo losowy).
- `optimization` jest praktyczne dla benchmarkow "best solution".
- domyslne `thin=1` i `burn_in=0` zostawiaja decyzje o diagnostyce na skrypt benchmarkowy.

### 5.2 Proposals

W benchmark runnerach (`run_pesplib.py`, `run_synthetic_benchmark.py`, `diagnostic_runner.py`) budowa propozycji jest zwykle:

- `SingleLineShift(sigma=5.0)` z waga `0.4`
- `UniformLineShift(max_shift=10)` z waga `0.1`
- `ComponentMove(sigma=5.0)` z waga `0.3` (opcjonalnie)
- `SmartRepair(sigma=2.0)` z waga `0.2` (opcjonalnie)

Dlaczego:
- miks laczy eksploracje globalna (single/uniform/component) z ruchem naprawczym (smart repair),
- `smart_repair` ma mniejsze `sigma` bo jest ruchem ukierunkowanym,
- `component` i `single_line` maja wieksze `sigma` dla szerszego przeszukiwania.

`ComponentMove`:
- jesli `p_decay` nie jest podane, auto: `min(0.5, 0.8 / d_avg)` na grafie zaleznosci.
- cel: utrzymac skonczywany, nie-eksplodujacy rozmiar komponentu.

### 5.3 Adaptacja (Tryb A vs Tryb B)

`src/mcmc/adaptation.py`:

- Tryb sampling:
  - `gamma_n = (n+1)^(-0.7)` (malejaca),
  - containment: `epsilon=0.05`, `sigma in [0.5, 50.0]`,
  - `target_accept=0.234`.
- Tryb optimization:
  - staly krok uczenia `0.01`,
  - bez containment clipping.

Dlaczego:
- w sampling wymusza to formalne warunki stabilnosci adaptacji,
- w optimization priorytetem jest szybsze strojenie heurystyczne.

### 5.4 Parallel Tempering

`src/mcmc/parallel_tempering.py`:

- `n_replicas=8`, `beta_min=0.1`, `beta_max=10.0`, `swap_interval=10`
- geometryczna drabina temperatur.

Dlaczego:
- geometria daje rownomierny rozklad "odleglosci temperatur" w skali log,
- `swap_interval=10` ogranicza koszt komunikacji przy zachowaniu regularnych prob swap.

## 6. Funkcja kosztu i jej defaulty

### 6.1 Transfer

`src/core/transfers.py` (`DEFAULT_TRANSFER_PARAMS`):

- `w_min=3.0`, `w_ideal=8.0`, `w_max=30.0`
- `alpha=2.0`, `beta=0.5`, `M=1e6`

Interpretacja:
- mocno karzemy naruszenia minimum (`M`),
- mocniej karzemy bycie "za wczesnie" niz "za pozno" (`alpha > beta`),
- waiting time liczony modulo czestotliwosci, zawsze nieujemny.

### 6.2 TotalCost

`src/cost/total.py`:

- domyslnie: `pax_weight=1.0`, `ops_weight=1.0`, `rob_weight=0.1`

Uwaga praktyczna:
- benchmarki (`run_pesplib.py`, `run_synthetic_benchmark.py`) nadpisuja zwykle na:
  - `pax_weight=1.0`, `ops_weight=0.1`, `rob_weight=0.05`
  - to celowo przesuwa nacisk na koszt pasazerski przy zachowaniu regularizacji OPS/ROB.

### 6.3 Robustness

`src/cost/robustness.py`:

- `DelayDistribution(rate=1/3, alpha=0.95)` -> srednie opoznienie 3 min, CVaR 95%.

Dlaczego:
- default daje umiarkowana "awersje do ogona" opoznien bez zdominowania glownych skladnikow kosztu.

## 7. Uruchamianie benchmarkow

### 7.1 Synthetic suite (zalecane do codziennej pracy)

Minimalnie:

```bash
python -m src.experiments.benchmark.run_synthetic_benchmark
```

Wlasna konfiguracja:

```bash
python -m src.experiments.benchmark.run_synthetic_benchmark \
  --config configs/synthetic_benchmark.yaml \
  --output results/synthetic_benchmark \
  --seed 42
```

Co jest zapisywane:

- `results.json` (pelne dane),
- `summary.csv` (zestawienie scenariuszy).

### 7.2 PESPlib z naszym modelem kosztu

```bash
python -m src.experiments.benchmark.run_pesplib --instance mini
python -m src.experiments.benchmark.run_pesplib --instance triangle --n_chains 4 --n_steps 25000
```

Domyslne CLI:

- `instance=mini`
- `n_chains=4`
- `n_steps=25000`
- `beta=1.0`
- `seed=42`

### 7.3 Raw PESPlib (oryginalny cel PESPlib)

```bash
python -m src.experiments.benchmark.run_pesplib_raw --instance R1L1
```

Domyslne CLI:

- `instance=R1L1`
- `n_steps=50000`
- `beta=0.0001`
- `seed=42`

Niskie `beta` wynika z innej skali celu (duze wartosci weighted slack).

## 8. Jak dobierac parametry w praktyce

Start:

1. `n_chains=4`, `n_steps=20k-25k`, `beta=1.0`
2. obserwuj `R-hat`, `ESS`, acceptance rate
3. dopiero potem strojenie proposal-i

Reguly:

- zbyt niska akceptacja (<0.1): zmniejsz `sigma` dla `single_line`/`component`
- zbyt wysoka akceptacja (>0.6) i slaby postep: zwieksz `sigma` lub podnies udzial `component`
- slaba mieszalnosc miedzy modami: dodaj kroki (`n_steps`) albo PT (`n_replicas`, drabina beta)
- jesli PT ma bardzo niskie swap rate, zaciesnij odstepy temperatur (czesto przez zmiane `beta_min/beta_max` lub liczby replik)

## 9. Testy: co odpalac i kiedy

### 9.1 Pelny pakiet

```bash
pytest tests/
```

### 9.2 Szybki smoke test MCMC + benchmark

```bash
pytest tests/test_mcmc/test_sampler.py
pytest tests/test_experiments/test_synthetic_benchmark.py
pytest tests/test_experiments/test_benchmark.py
```

### 9.3 Testy scenariuszy instancyjnych

```bash
pytest tests/test_io/test_synthetic_scenarios.py
```

To pokrywa m.in.:
- wszystkie typy scenariuszy z `SyntheticScenarioFactory`,
- poziomy trudnosci,
- podstawowa spojnosc struktury sieci i transferow.

## 10. Reprodukowalnosc

Zasady:

- ustawiaj `--seed` w skryptach CLI,
- multi-chain uzywa zwykle `base_seed + chain_id`,
- dla suite syntetycznej scenariusze dostaja przesuniecie seed-a o krokach (`base_seed + i*1000`).

Dzieki temu porownania miedzy runami sa stabilne.

## 11. Najczestsze problemy

- `Config not found` w synthetic benchmark:
  - sprawdz `--config` i sciezke robocza.
- `Instance not found` w PESPlib:
  - `run_pesplib`: szuka w `data/pesplib/raw|converted|synthetic`
  - `run_pesplib_raw`: oczekuje pliku `data/pesplib/raw/<INSTANCE>.txt`.
- Diagnostyka nie liczy `R-hat`:
  - potrzebujesz co najmniej 2 chain (`n_chains >= 2`).

## 12. Pliki, ktore warto znac

- `src/mcmc/sampler.py` - petla MH
- `src/mcmc/proposals/*.py` - kernele propozycji
- `src/mcmc/adaptation.py` - adaptacja wag i skali
- `src/mcmc/parallel_tempering.py` - replica exchange + inertia
- `src/experiments/benchmark/run_synthetic_benchmark.py` - benchmarki syntetyczne
- `src/experiments/benchmark/run_pesplib.py` - benchmark na mini/triangle + loader PESPlib
- `src/experiments/benchmark/run_pesplib_raw.py` - benchmark na raw PESPlib objective
- `tests/test_experiments/*.py` i `tests/test_io/test_synthetic_scenarios.py` - testy instancji i runnerow

