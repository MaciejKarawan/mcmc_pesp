# Plan eksperymentów MCMC-PESP

## Przegląd

Ten dokument zawiera szczegółowy plan eksperymentów do przeprowadzenia w celu walidacji i optymalizacji algorytmu MCMC-PESP. Wszystkie pytania mają charakter empiryczny i nie mogą być rozstrzygnięte teoretycznie.

---

## 1. Mixing time i zbieżność

### 1.1 Podstawowa zbieżność

**Pytanie**: Ile kroków potrzeba do zbieżności?

**Setup**:
- Instancja: PESPlib `toy` (najmniejsza)
- Łańcuchy: 4 niezależne, różne punkty startowe (losowe)
- Kroki: 100k, 500k, 1M
- Propozycja: Single Line Shift (baseline)

**Metryki**:
- Gelman-Rubin $\hat{R}$ dla `Cost(x)`
- Gelman-Rubin $\hat{R}$ dla `avg_transfer_time`
- Gelman-Rubin $\hat{R}$ dla `worst_transfer`

**Kryterium sukcesu**: $\hat{R} < 1.1$ dla wszystkich metryk

**Kod**:
```bash
python -m src.experiments.analysis.convergence \
    --instance toy \
    --n_chains 4 \
    --n_steps 100000 500000 1000000 \
    --proposal single_line
```

### 1.2 Efektywna wielkość próby

**Pytanie**: Jaki ESS / krok?

**Setup**: Jak wyżej

**Metryki**:
- ESS dla `Cost(x)`
- ESS / 1000 kroków
- Czas autokorelacji

**Kryterium sukcesu**: ESS/krok > 0.01

### 1.3 Ucieczka z minimów lokalnych

**Pytanie**: Czy łańcuch eksploruje wiele basenów atrakcji?

**Setup**:
- Instancja: PESPlib `small` (wyraźna multimodalność)
- Długi łańcuch: 1M kroków
- Zapisuj `Cost(x)` co 100 kroków

**Analiza**:
- Trace plot — szukaj "skoków" między poziomami
- Histogram `Cost(x)` — czy multimodalny?
- Liczba unikalnych "basenów" (clustering)

### 1.4 Skalowanie mixing time

**Pytanie**: Jak mixing time zależy od rozmiaru sieci?

**Setup**:
- Instancje: `toy`, `small`, `medium`, `large` z PESPlib
- Dla każdej: zmierz czas do $\hat{R} < 1.1$

**Analiza**:
- Plot: mixing_time vs |L| (liczba linii)
- Dopasuj: $t_{mix} \propto |L|^\alpha$
- Oceń $\alpha$

---

## 2. Porównanie propozycji ruchów

### 2.1 ESS per propozycja

**Pytanie**: Która propozycja ma najlepszy ESS/krok?

**Setup**:
- Instancja: PESPlib `small`
- Propozycje testowane osobno:
  - Single Line Shift
  - Component Move (p_decay = 0.3)
  - Smart Repair
  - Swap
- Dla każdej: 100k kroków, 4 łańcuchy

**Metryki**:
- ESS / krok
- ESS / sekunda CPU
- Acceptance rate

**Wizualizacja**: Bar chart porównawczy

### 2.2 Optymalny mix propozycji

**Pytanie**: Jakie wagi $p_j$ dają najlepsze wyniki?

**Setup**:
- Instancja: PESPlib `small`
- Grid search po wagach:
  ```
  single: [0.2, 0.4, 0.6]
  component: [0.1, 0.2, 0.3]
  repair: [0.1, 0.2, 0.3]
  swap: [0.05, 0.1, 0.15]
  ```
- Dla każdej kombinacji: 200k kroków

**Metryki**:
- Końcowy `Cost(x_best)`
- Czas do zbieżności ($\hat{R} < 1.1$)

### 2.3 Smart Repair — czy przyspiesza?

**Pytanie**: Czy Smart Repair istotnie poprawia zbieżność?

**Setup**:
- Instancja: PESPlib `medium`
- Warianty:
  - A: mix bez Smart Repair
  - B: mix z Smart Repair (20%)
- Dla każdego: 500k kroków, 4 łańcuchy

**Test statystyczny**: t-test na końcowy koszt między A i B

### 2.4 Component Move — optymalny p_decay

**Pytanie**: Jaki p_decay jest najlepszy?

**Setup**:
- Instancja: PESPlib `small`
- Testowane wartości: p_decay ∈ {0.1, 0.2, 0.3, 0.5, 0.7}
- Dla każdej: 100k kroków

**Metryki**:
- Średni rozmiar komponentu
- ESS / krok
- Acceptance rate

**Oczekiwanie**: Optimum przy p_decay ≈ 0.8 / d_avg

---

## 3. Parallel Tempering

### 3.1 Optymalna liczba replik

**Pytanie**: Ile replik daje najlepszy trade-off jakość/koszt?

**Setup**:
- Instancja: PESPlib `medium`
- Testowane: M ∈ {4, 8, 16, 32}
- β_min = 0.1, β_max = 10.0
- Dla każdego M: 500k kroków (łącznie, podzielone na repliki)

**Metryki**:
- Końcowy `Cost(x_best)`
- Czas CPU do znalezienia najlepszego rozwiązania
- Swap acceptance rate między replikami

### 3.2 Schemat temperatur

**Pytanie**: Geometryczny vs adaptacyjny rozstaw temperatur?

**Setup**:
- Instancja: PESPlib `medium`
- Warianty:
  - A: Geometryczny (β_i = β_1 * r^(i-1))
  - B: Adaptacyjny (dostosowuj aby swap rate ≈ 25%)
- M = 16, 500k kroków

**Metryki**:
- Swap acceptance rate per para replik
- Końcowa jakość

### 3.3 Inercja vs zwykłe PT

**Pytanie**: Czy inercja hierarchiczna pomaga?

**Setup**:
- Instancja: PESPlib `medium` (z wyraźną hierarchią modalności)
- Warianty:
  - A: Zwykłe PT (I_L = 1 dla wszystkich)
  - B: Inercyjne PT (I_IC = 20, I_Regio = 5, I_Local = 1)
- M = 16, 500k kroków

**Metryki**:
- Końcowy `Cost(x_best)`
- Stabilność między uruchomieniami (std dev)

### 3.4 Propagacja dobrych rozwiązań

**Pytanie**: Jak szybko dobre rozwiązania z gorących replik trafiają do zimnych?

**Setup**:
- Instancja: PESPlib `small`
- M = 8
- Śledź: dla każdego stanu w replice M (najzimniejsza), skąd "pochodzi" (która replika go "odkryła")

**Analiza**:
- Histogram "wieku" stanów w zimnej replice
- Średni czas propagacji od gorącej do zimnej

---

## 4. Hierarchia

### 4.1 Hierarchia vs płaska optymalizacja

**Pytanie**: Czy hierarchia w ogóle pomaga?

**Setup**:
- Instancja: Syntetyczna sieć z 3 poziomami (IC + Regio + Local)
- Warianty:
  - A: Bez hierarchii (wszystkie linie równe)
  - B: Top-down (zamrażanie)
  - C: Inercyjne PT
- 500k kroków każdy

**Metryki**:
- Końcowy `Cost_total`
- `Cost_PAX` per poziom hierarchii
- Czas obliczeń

### 4.2 Optymalne wartości inercji

**Pytanie**: Jakie I_L są najlepsze?

**Setup**:
- Instancja: Syntetyczna 3-poziomowa
- Grid search:
  ```
  I_IC ∈ {5, 10, 20, 50}
  I_Regio ∈ {1, 2, 5, 10}
  I_Local = 1 (bazowy)
  ```

**Metryki**:
- Końcowy koszt
- Stabilność hierarchii (jak często IC się zmienia w zimnych replikach)

---

## 5. Kalibracja funkcji kosztu

### 5.1 Wrażliwość na M (bariera)

**Pytanie**: Jak wybór M wpływa na wyniki?

**Setup**:
- Instancja: PESPlib `small`
- Testowane: M ∈ {10³, 10⁴, 10⁵, 10⁶, 10⁷}
- 200k kroków dla każdego

**Metryki**:
- % stanów naruszających ograniczenia
- Końcowy koszt (bez składnika bariery)
- Acceptance rate

**Oczekiwanie**: Za małe M → dużo naruszeń; za duże M → niski acceptance rate

### 5.2 Trade-off PAX vs OPS vs ROB

**Pytanie**: Jak dobrać wagi λ?

**Setup**:
- Instancja: PESPlib `medium` z danymi o obrotach
- Grid search:
  ```
  λ_pax = 1.0 (bazowy)
  λ_ops ∈ {0.1, 1.0, 10.0}
  λ_rob ∈ {0.01, 0.1, 1.0}
  ```

**Analiza**:
- Krzywa Pareto: Cost_PAX vs Cost_OPS
- Krzywa Pareto: Cost_PAX vs Cost_ROB

### 5.3 CVaR vs wartość oczekiwana

**Pytanie**: Czy CVaR daje bardziej robustne rozwiązania?

**Setup**:
- Instancja: PESPlib `small`
- Warianty:
  - A: Robustność przez E[L]
  - B: Robustność przez CVaR_0.95[L]
- Po optymalizacji: symulacja 1000 scenariuszy opóźnień

**Metryki**:
- % utraconych przesiadek w symulacji
- 95-percentyl czasu podróży

---

## 6. Benchmark vs alternatywy

### 6.1 Baseline: Random Search

**Setup**:
- Instancje: PESPlib `toy`, `small`, `medium`
- Random search: N ∈ {1k, 10k, 100k, 1M} losowych stanów
- Zwróć najlepszy

**Metryki**:
- Końcowy koszt vs N
- Czas CPU

**Cel**: Ustalić minimum, które MCMC musi pobić

### 6.2 Baseline: Greedy Local Search

**Setup**:
- Instancje: jak wyżej
- Algorytm:
  1. Zacznij od losowego stanu
  2. Powtarzaj: znajdź najgorsze ograniczenie, napraw je optymalnie
  3. Stop gdy brak poprawy

**Metryki**:
- Końcowy koszt
- Liczba iteracji do zbieżności

### 6.3 Simulated Annealing

**Pytanie**: Czy MCMC (PT) bije SA z tymi samymi propozycjami?

**Setup**:
- Instancje: PESPlib `small`, `medium`
- SA: te same propozycje co MCMC, schedule chłodzenia (liniowy, geometryczny)
- MCMC: Parallel Tempering, M = 16
- Ten sam budżet kroków

**Metryki**:
- Końcowy koszt
- Czas do najlepszego rozwiązania
- Stabilność (std dev między uruchomieniami)

### 6.4 Porównanie z MIP (małe instancje)

**Pytanie**: Jak daleko od optimum globalnego?

**Setup**:
- Instancje: PESPlib `toy`, `small` (gdzie MIP jest wykonalny)
- MIP solver: Gurobi/CPLEX z limitem czasu 1h
- MCMC: 1M kroków

**Metryki**:
- Gap: (MCMC_cost - MIP_opt) / MIP_opt
- Czas do rozwiązania

### 6.5 Porównanie z publikowanymi wynikami

**Pytanie**: Jak wypadamy vs state-of-the-art?

**Setup**:
- Instancje: PESPlib z opublikowanymi wynikami
- Zbierz wyniki z literatury (SAT, MIP, GA)
- Uruchom MCMC

**Metryki**:
- Ranking wśród metod
- Gap do najlepszego znanego

---

## 7. Walidacja na rzeczywistych danych

### 7.1 PESPlib — pełna walidacja

**Setup**:
- Wszystkie instancje PESPlib z known optimum
- MCMC: najlepsze ustawienia z eksperymentów 1-6
- 3 uruchomienia per instancja

**Raport**:
| Instancja | |V| | |E| | Optimum | MCMC best | MCMC avg | Gap |
|-----------|-----|-----|---------|-----------|----------|-----|

### 7.2 GTFS — rzeczywista sieć

**Setup**:
- Dane: ZTM Warszawa lub PKP
- Konwersja GTFS → nasz format
- Uruchom MCMC

**Analiza**:
- Czy rozwiązanie ma sens? (ekspert review)
- Porównanie z obecnym rozkładem

### 7.3 Identyfikacja wąskich gardeł

**Pytanie**: Czy model prawidłowo identyfikuje problematyczne przesiadki?

**Setup**:
- Instancja: GTFS rzeczywista
- Analiza: top 10 przesiadek o najwyższym koszcie

**Walidacja**: Czy eksperci zgadzają się, że to są problemy?

---

## 8. Skalowanie

### 8.1 Czas vs rozmiar sieci

**Setup**:
- Instancje: PESPlib od `toy` do `large`
- Mierz: czas 10k kroków

**Analiza**:
- Plot: czas vs |L|
- Dopasuj: t ∝ |L|^α

### 8.2 Inkrementalny koszt — speedup

**Setup**:
- Instancja: PESPlib `medium`
- Warianty:
  - A: Naiwne obliczenie Cost(x) przy każdym kroku
  - B: Inkrementalne delta_cost()

**Metryki**:
- Czas per krok
- Speedup factor

### 8.3 Profiling

**Setup**:
- Instancja: PESPlib `large`
- Narzędzie: cProfile / py-spy

**Analiza**:
- % czasu w każdej funkcji
- Identyfikacja wąskich gardeł

---

## Harmonogram eksperymentów

| Tydzień | Eksperymenty | Zależności |
|---------|--------------|------------|
| 3 | 1.1, 1.2 (podstawowa zbieżność) | Faza 2 implementacji |
| 4 | 2.1, 2.2 (propozycje) | 1.* |
| 5 | 3.1, 3.2, 3.3 (PT) | 2.* |
| 6 | 6.1-6.4 (benchmark) | 3.* |
| 7 | 4.*, 5.* (hierarchia, kalibracja) | 6.* |
| 8 | 7.*, 8.* (walidacja, skalowanie) | wszystko |

---

## Raportowanie wyników

Każdy eksperyment powinien produkować:

1. **JSON z surowymi danymi**: `results/<exp_id>/raw_data.json`
2. **Podsumowanie**: `results/<exp_id>/summary.md`
3. **Wykresy**: `results/<exp_id>/plots/`
4. **Notebook z analizą**: `notebooks/<exp_id>_analysis.ipynb`

Format summary.md:
```markdown
# Eksperyment: <nazwa>

## Setup
- Instancja: ...
- Parametry: ...

## Wyniki
| Metryka | Wartość |
|---------|---------|
| ... | ... |

## Wnioski
...

## Następne kroki
...
```
