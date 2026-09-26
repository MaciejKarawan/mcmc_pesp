# MCMC-PESP

Implementacja algorytmu MCMC (Markov Chain Monte Carlo) do rozwiązywania problemu PESP (Periodic Event Scheduling Problem) — optymalizacji rozkładów jazdy transportu publicznego.

## Cechy

- **Miękkie ograniczenia** — bariery zamiast twardych ograniczeń, gwarantujące ergodyczność
- **Strukturalne propozycje ruchów** — Component Move, Smart Repair
- **Inercyjny Parallel Tempering** — hierarchiczna optymalizacja bez twardego zamrażania
- **Robustność CVaR** — optymalizacja pod kątem najgorszych scenariuszy opóźnień
- **Walidacja PESPlib** — porównanie z benchmarkami i znanymi optimami

## Instalacja

```bash
git clone https://github.com/MaciejKarawan/mcmc_pesp.git
cd mcmc_pesp
pip install -r requirements.txt
```

## Szybki start

```bash
# Uruchom PoC na syntetycznej sieci
python -m src.experiments.poc.triangle

# Benchmark na PESPlib
python -m src.experiments.benchmark.run --instance small
```

## Dokumentacja

- [CLAUDE.md](CLAUDE.md) — instrukcje dla deweloperów / Claude Code
- [docs/MATH_SPEC.md](docs/MATH_SPEC.md) — pełna specyfikacja matematyczna
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — plan eksperymentów


## Struktura projektu

```
mcmc-pesp/
├── src/
│   ├── core/           # Struktury danych (Network, Schedule, Transfer)
│   ├── cost/           # Funkcje kosztu
│   ├── mcmc/           # Algorytmy MCMC i propozycje ruchów
│   ├── metrics/        # Metryki jakości i diagnostyka
│   ├── io/             # Loadery (PESPlib, GTFS, synthetic)
│   └── experiments/    # Skrypty eksperymentów
├── tests/              # Testy jednostkowe
├── data/               # Dane (PESPlib, GTFS)
├── configs/            # Konfiguracje
└── notebooks/          # Jupyter notebooks
```

## Licencja

MIT
