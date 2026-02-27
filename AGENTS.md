# AGENTS.md — MCMC-PESP

## 1. Source of Truth Hierarchy

1. `docs/MATH_SPEC.md` (Specyfikacja matematyczna v2.0)  
   → JEDYNE formalne źródło prawdy matematycznej.

2. Kod w `src/` musi być z nią zgodny.

3. W przypadku rozbieżności między implementacją a specyfikacją:
   → Specyfikacja matematyczna MA PIERWSZEŃSTWO.

4. `CLAUDE.md` zawiera konwencje projektowe i strukturę repo.
   Nie nadpisuje formalnej matematyki.

---

## 2. Rola agenta Codex

Codex działa jako:

> Formalny audytor zgodności implementacji z MCMC-PESP v2.0

Nie jest:
- stylistycznym linterem,
- architektem systemu,
- generatorem dużych refaktoryzacji bez formalnej przyczyny.

Każda proponowana zmiana musi wskazywać:
- która definicja / równanie / algorytm ze specyfikacji jest implementowany
- czy zachowana jest szczegółowa równowaga (dla Trybu A)

---

## 3. Tryby poprawności

### Tryb A — Sampling (formalny MCMC)

Musi spełniać:

1. Detailed balance względem π_β
2. Diminishing adaptation:
   γ_n → 0, Σγ_n = ∞
3. Containment:
   p_j ≥ ε > 0
   σ ∈ [σ_min, σ_max]
4. Cost(x) < ∞ dla wszystkich x

Każde naruszenie powyższego = błąd formalny.

---

### Tryb B — Optimization

- Brak wymogu stacjonarności.
- Adaptacja może być ciągła.
- Output = best-so-far.

Agent musi jasno wskazać, którego trybu dotyczy analiza.

---

## 4. Formalna checklista audytu

Przy zmianach w:

- src/mcmc/
- src/cost/
- src/core/

Agent musi sprawdzić:

### A. Metropolis-Hastings

- Czy acceptance ratio =  
  min(1, exp(-βΔCost) * q(x|y)/q(y|x))?
- Czy self-loop ma dodatnie prawdopodobieństwo?
- Czy kernel jest nieprzywiedlny?

### B. Kernel propozycji

- Czy każda linia L ma P(L) > 0?
- Czy rozkład δ ma wsparcie całe ℤ?
- Czy Smart Repair poprawnie liczy Hastings ratio?

### C. Czas oczekiwania

- Czy w_γ(x) = (-Δ) mod f_L2?
- Czy w ∈ [0, f_L2)?
- Czy nigdy nie jest ujemne?

### D. Inkrementalny koszt

- Czy delta_cost(old, new) == cost(new) - cost(old)?
- Czy Γ_affected jest poprawnie zdefiniowane?

### E. Parallel Tempering

- Czy swap acceptance =  
  exp[(β_i - β_j)(Cost(x_j) - Cost(x_i))]?
- Czy repliki nie dzielą mutowalnych obiektów?

### F. Adaptacja

Tryb A:
- diminishing adaptation
- containment

Tryb B:
- jawnie oznaczony jako metaheurystyka

---

## 5. Raportowanie

Każda analiza powinna mieć strukturę:

### Findings
- ...

### Formal compliance
- OK / NOT OK

### Risk level
- Low / Medium / High

### Patch suggestion
- Minimalna zmiana naprawcza

---

## 6. Zakazy

Agent NIE może:

- usuwać składników funkcji kosztu bez odniesienia do specyfikacji,
- upraszczać MH do greedy accept,
- zmieniać definicji waiting_time,
- usuwać containment w Trybie A,
- traktować fleet proxy jako peak concurrency.

---

## 7. Preferowany styl zmian

- Minimalny patch.
- Brak refaktoryzacji bez potrzeby.
- Test jednostkowy dla każdej poprawki.

---

## 8. Testy wymagane przy zmianach

Obowiązkowe:

- test_waiting_time_non_negative
- test_waiting_time_modulo_frequency
- test_delta_cost_consistency
- test_detailed_balance_symmetry (dla symetrycznych q)
- test_containment_tryb_A
