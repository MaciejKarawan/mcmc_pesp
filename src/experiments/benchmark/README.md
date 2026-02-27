# PESPlib Benchmark

Ten folder zawiera skrypty do uruchamiania benchmarków MCMC na instancjach PESPlib.

## Skrypty

W folderze znajdują się dwa skrypty benchmarkowe, różniące się podejściem do funkcji celu:

| Skrypt | Funkcja celu | Instancje | Przeznaczenie |
|--------|--------------|-----------|---------------|
| `run_pesplib.py` | TotalCost (PAX + OPS + ROB) | syntetyczne (mini, triangle) | Testowanie naszego modelu |
| `run_pesplib_raw.py` | Oryginalny cel PESPlib (weighted slack) | R1L1 - R4L4 | Porównanie z benchmarkiem |

### run_pesplib.py

Główny skrypt benchmarku wykorzystujący **nasz model kosztów** (TotalCost):

- **Koszt pasażerski (PAX)**: miękkie ograniczenia na czasy przesiadek
- **Koszt operacyjny (OPS)**: obroty taboru (turnaround)
- **Koszt robustności (ROB)**: CVaR dla opóźnień

Obsługuje wiele propozycji MCMC (SingleLineShift, ComponentMove, SmartRepairMove) oraz multi-chain diagnostics (R-hat, ESS).

**Typowe użycie:**
```bash
# Syntetyczna instancja mini (10 zdarzeń)
python -m src.experiments.benchmark.run_pesplib --instance mini

# Z wieloma łańcuchami dla diagnostyki zbieżności
python -m src.experiments.benchmark.run_pesplib --instance mini --n_chains 4

# Więcej kroków dla lepszej zbieżności
python -m src.experiments.benchmark.run_pesplib --instance triangle --n_steps 50000
```

### run_pesplib_raw.py

Skrypt do testowania na **oryginalnych instancjach PESPlib** (Set-01) z ich natywnym celem optymalizacji (weighted slack sum). Używa pojedynczych zdarzeń jako zmiennych decyzyjnych (nie offsetów linii).

**Typowe użycie:**
```bash
# Najmniejsza instancja R1L1
python -m src.experiments.benchmark.run_pesplib_raw --instance R1L1

# Więcej kroków i niższa temperatura
python -m src.experiments.benchmark.run_pesplib_raw --instance R1L1 --n_steps 100000 --beta 0.0001
```

---

## Parametry CLI

### run_pesplib.py

| Parametr | Domyślna | Opis |
|----------|----------|------|
| `--instance` | `mini` | Nazwa instancji. Dostępne: `mini`, `triangle`, lub ścieżka do instancji PESPlib |
| `--n_chains` | `4` | Liczba niezależnych łańcuchów MCMC (dla R-hat) |
| `--n_steps` | `25000` | Liczba kroków MCMC na łańcuch |
| `--beta` | `1.0` | Odwrotność temperatury (wyższe = bardziej zachłanny) |
| `--seed` | `42` | Ziarno losowości |
| `--output` | `None` | Ścieżka do pliku JSON z wynikami (domyślnie stdout) |

### run_pesplib_raw.py

| Parametr | Domyślna | Opis |
|----------|----------|------|
| `--instance` | `R1L1` | Nazwa instancji PESPlib (R1L1 - R4L4) |
| `--n_steps` | `50000` | Liczba kroków MCMC |
| `--beta` | `0.0001` | Odwrotność temperatury (mała ze względu na duże wartości celu) |
| `--seed` | `42` | Ziarno losowości |
| `--output` | `None` | Ścieżka do pliku JSON z wynikami |

---

## Rekomendowane parametry

Parametry zalecane na podstawie specyfikacji matematycznej (docs/MATH_SPEC.md):

### Wagi propozycji

| Propozycja | Waga | Opis |
|------------|------|------|
| SingleLineShift | 0.4 | Przesunięcie pojedynczej linii (σ ~ N(0, σ²)) |
| ComponentMove | 0.3 | Przesunięcie skorelowanych linii (BFS z zanikiem) |
| SmartRepairMove | 0.2 | Celowana naprawa naruszonego ograniczenia |
| UniformLineShift/Swap | 0.1 | Dywersyfikacja |

### Parametry algorytmu

| Parametr | Wartość | Źródło |
|----------|---------|--------|
| `beta` | 1.0 | MATH_SPEC.md sekcja 5.1 |
| `n_chains` | 4 | Minimum dla R-hat (sekcja 8.1) |
| `p_decay` (ComponentMove) | ~0.8/d_avg | Sekcja 6.2 - zapewnia skończony rozmiar komponentu |
| `sigma` (shift) | 5.0 | Dostrojone empirycznie |
| `sigma_small` (SmartRepair) | 2.0 | Mały szum wokół naprawy |

### Wagi kosztów (TotalCost)

| Waga | Wartość | Opis |
|------|---------|------|
| `pax_weight` | 1.0 | Koszt pasażerski (główny) |
| `ops_weight` | 0.1 | Koszt operacyjny (turnaround) |
| `rob_weight` | 0.05 | Robustność (CVaR) |

### CVaR

| Parametr | Wartość | Opis |
|----------|---------|------|
| `alpha` | 0.95 | Poziom ufności dla CVaR (95. percentyl) |
| `rate` (exp) | 1/3 | Średnie opóźnienie = 3 minuty |

---

## Kryteria sukcesu

Benchmark uznajemy za udany gdy spełnione są następujące kryteria:

### 1. Zbieżność MCMC

| Metryka | Kryterium | Źródło |
|---------|-----------|--------|
| R-hat (Gelman-Rubin) | < 1.1 | MATH_SPEC.md sekcja 8.1 |
| ESS (Effective Sample Size) | > 100 | MATH_SPEC.md sekcja 8.2 |

**R-hat** mierzy zgodność między łańcuchami. Wartość bliska 1 oznacza, że wszystkie łańcuchy eksplorują tę samą dystrybucję.

**ESS** mierzy efektywną liczbę niezależnych próbek (uwzględniając autokorelację). ESS > 100 zapewnia wiarygodne estymaty.

### 2. Jakość rozwiązania

| Metryka | Kryterium | Opis |
|---------|-----------|------|
| Gap do optimum | ≤ 10% | Dla znanych instancji PESPlib |
| MissRate | = 0% | Brak przesiadek z czasem < w_min |
| Feasibility | 100% | Wszystkie ograniczenia spełnione |

**Gap** definiowany jako:
```
Gap = (nasz_wynik / known_optimum - 1) × 100%
```

### 3. Acceptance rate

| Propozycja | Optymalny zakres | Opis |
|------------|------------------|------|
| SingleLineShift | 20-40% | Zbyt niski = za duże skoki |
| ComponentMove | 15-30% | Naturalnie niższy (większe zmiany) |
| SmartRepairMove | 25-50% | Celowane naprawy |

---

## Format wyjściowy (JSON)

```json
{
  "instance": "mini",
  "n_chains": 4,
  "n_steps_per_chain": 25000,
  "best_solution": {
    "offsets": {"L1_>": 0, "L2_>": 15, "L3_>": 30},
    "our_cost": 45.67,
    "pesplib_objective": 12345
  },
  "diagnostics": {
    "r_hat": 1.02,
    "ess_total": 3440,
    "ess_per_chain": [860, 890, 850, 840],
    "converged": true
  },
  "per_chain_results": [
    {
      "chain_id": 0,
      "best_cost": 46.2,
      "acceptance_rate": 0.28,
      "n_samples": 2250
    }
  ]
}
```

---

## Przykłady użycia

### Szybki test na syntetycznej instancji

```bash
python -m src.experiments.benchmark.run_pesplib --instance mini --n_steps 10000
```

### Pełny benchmark z diagnostyką

```bash
python -m src.experiments.benchmark.run_pesplib \
    --instance triangle \
    --n_chains 4 \
    --n_steps 50000 \
    --output results/triangle_benchmark.json
```

### Porównanie z PESPlib optimum

```bash
python -m src.experiments.benchmark.run_pesplib_raw \
    --instance R1L1 \
    --n_steps 100000 \
    --output results/r1l1_benchmark.json
```

### Batch benchmark (wszystkie instancje R1)

```bash
for inst in R1L1 R1L2 R1L3 R1L4; do
    python -m src.experiments.benchmark.run_pesplib_raw \
        --instance $inst \
        --n_steps 50000 \
        --output results/${inst}.json
done
```

---

## Znane instancje PESPlib (Set-01)

| Instancja | Znane optimum | Lower bound | Rozmiar |
|-----------|---------------|-------------|---------|
| R1L1 | 29,894,745 | 20,230,655 | mała |
| R1L2 | 31,081,380 | 23,093,710 | mała |
| R1L3 | 35,697,890 | 28,135,880 | mała |
| R1L4 | 54,104,635 | 47,227,510 | mała |
| R2L1 | 42,161,450 | 26,316,195 | średnia |
| R2L2 | 43,506,390 | 29,936,230 | średnia |
| R2L3 | 49,961,540 | 39,215,050 | średnia |
| R2L4 | 115,627,335 | 87,549,645 | średnia |
| R3L1 | 59,408,020 | 33,890,315 | duża |
| R3L2 | 61,346,440 | 42,207,825 | duża |
| R3L3 | 71,877,820 | 55,082,680 | duża |
| R3L4 | 172,251,580 | 126,685,195 | duża |
| R4L1 | 75,588,190 | 52,310,810 | bardzo duża |
| R4L2 | 77,627,590 | 49,862,760 | bardzo duża |
| R4L3 | 97,006,160 | 66,371,585 | bardzo duża |
| R4L4 | 233,312,290 | 145,976,695 | bardzo duża |

Źródło: https://num.math.uni-goettingen.de/~m.goerigk/pesplib/

---

## Rozwiązywanie problemów

### "Instance not found"

Upewnij się, że instancja istnieje:
- Syntetyczne: `mini`, `triangle` (generowane automatycznie)
- PESPlib raw: pobierz pliki .txt do `data/pesplib/raw/`

### Niska acceptance rate (< 10%)

- Zmniejsz `sigma` w propozycjach
- Zmniejsz `beta` (wyższa temperatura)

### R-hat > 1.2

- Zwiększ `n_steps` (więcej burn-in)
- Uruchom więcej łańcuchów
- Sprawdź czy problem nie ma wielu silnych minimów lokalnych

### ESS < 100

- Zwiększ `n_steps`
- Zmniejsz `thin` (mniej agresywne proreżanie)
- Dodaj propozycje dywersyfikujące (swap, component move)
