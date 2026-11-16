# MCMC-PESP: Markov Chain Monte Carlo for Public-Transport Timetabling

**Research framework that applies MCMC to the Periodic Event Scheduling Problem (PESP), i.e. choosing line offsets in a periodic timetable so that passenger transfers work. It uses soft constraints, structured proposal moves and parallel tempering with inertia. The code is tested on synthetic networks, PESPlib benchmarks and GTFS data of a Polish regional railway (Koleje Dolnośląskie).**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-423%20passing-brightgreen)
![Topic](https://img.shields.io/badge/topic-MCMC%20%7C%20operations%20research-blueviolet)

## The problem

A periodic timetable is determined by the **offset** of each line within a common period (T = 1440 min). Shifting one line changes waiting times at every transfer it takes part in, so the cost landscape is highly multimodal and has sharp discontinuities: a transfer waits 0 minutes or a full headway. Classical PESP uses hard constraints and integer programming. This project instead treats the timetable as a **state of a Markov chain** and explores the space with Metropolis–Hastings.

## Approach

- **Soft constraints**: quadratic barriers instead of infeasible = +∞. The chain therefore stays ergodic and can cross infeasible regions
- **Cost** = passenger transfer cost + operational cost (turnaround, fleet-intensity proxy via Little's law) + a CVaR-based **robustness proxy** for delays
- **Proposal moves**
  - *Single Line Shift*: random offset change of one line
  - *Component Move*: correlated shift of a connected group of lines with decay
  - *Smart Repair*: targeted fix of the worst transfer, using the minimal-norm shift and discrete proposal masses so that the Hastings ratio stays exact
- **Two modes**: *sampling* (formally valid MCMC with diminishing adaptation, γₙ = n^−0.7, and containment pⱼ ≥ ε) and *optimization* (best-so-far metaheuristic)
- **Parallel tempering with inertia**: replicas at different temperatures exchange states, and inertia weights damp moves of "heavy" lines
- **Diagnostics**: Gelman–Rubin R̂, effective sample size, acceptance and swap rates, peak vehicle concurrency (sweep-line)

## Benchmarks (quick synthetic suite, 5,000 cost evaluations, seed 42)

Best cost found (lower is better):

| Scenario | Random search | MCMC (single chain) | Simulated annealing | Parallel tempering (M = 4) |
|---|---|---|---|---|
| Hub, easy | 78.39 | 3.144 | 3.144 | **3.144** |
| Chain, easy | 0.699 | 0.699 | 0.699 | 0.699 |
| Multi-hub mesh, easy | 23,173 | 8,961 | **3,114** | 3,292 |

Source: [`results/benchmark/comparison_quick.json`](results/benchmark/comparison_quick.json). More runs (PT calibration, hard instances, GTFS pilots) are in [`results/`](results/).

## Repository structure

```
src/core/           Network, Line, Station, Schedule (offsets mod T), Transfer
src/cost/           transfer cost with barriers, operational, robustness proxy, total + Δ-cost
src/mcmc/           sampler (MH loop), proposals/, parallel_tempering, adaptation
src/metrics/        convergence (R̂, ESS), peak_concurrency
src/io/             PESPlib loaders, GTFS loader, synthetic scenario factory
src/experiments/    poc/ (triangle), benchmark/ (synthetic, PESPlib, GTFS A/B), analysis/
configs/            YAML experiment configs
kd_transit/         GTFS feed of Koleje Dolnośląskie
notebooks/          PoC, diagnostics, cost calibration, benchmark comparison, profiling
docs/               MATH_SPEC.md (formal specification), EXPERIMENTS.md, user guide (PL)
tests/              423 unit and integration tests
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q

# proof of concept on a small synthetic network
python -m src.experiments.poc.triangle

# synthetic benchmark suite (see --help for scenarios and budgets)
python -m src.experiments.benchmark.run_synthetic_benchmark --help
```

The formal model is in [`docs/MATH_SPEC.md`](docs/MATH_SPEC.md) and the full mathematical specification in [`mcmc_pesp_mathematical_specification.md`](mcmc_pesp_mathematical_specification.md). Documentation is partly in Polish.

## Status

Phases 1–3 are complete: core model, cost functions, proposal moves, formally valid sampling mode and parallel tempering. Current work covers scaling (incremental cost, profiling), large PESPlib instances and real GTFS data.

## License

MIT
