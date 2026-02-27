# Metody MCMC dla Problemu Periodic Event Scheduling (PESP)

## Dokument matematyczny — Wersja 2.0 (skonsolidowana)

**Projekt MCMC-PESP** | Luty 2026

---

## Streszczenie

Niniejszy dokument przedstawia pełną specyfikację matematyczną podejścia opartego na metodach Monte Carlo łańcuchów Markowa (MCMC) do rozwiązywania problemu PESP (Periodic Event Scheduling Problem) w kontekście optymalizacji rozkładów jazdy transportu publicznego.

Model wykorzystuje:
- **Miękkie ograniczenia** (soft constraints) z funkcjami barier — gwarantując ergodyczność łańcucha Markowa
- **Ograniczenia zasobowe** (obroty taboru, proxy intensywności floty)
- **Strukturalne propozycje ruchów** (Component Move, Smart Repair) redukujące czas mieszania
- **Inercyjny Parallel Tempering** zastępujący sztywną hierarchię top-down
- **Heurystyczne proxy robustności** oparte na ogonie rozkładu opóźnień

Algorytm jest specyfikowany w dwóch trybach: *Tryb A* (formalny MCMC z próbkowaniem z rozkładu stacjonarnego) oraz *Tryb B* (metaheurystyka optymalizacyjna inspirowana MCMC).

---

## Spis treści

1. [Wprowadzenie i motywacja](#1-wprowadzenie-i-motywacja)
2. [Model sieci transportowej](#2-model-sieci-transportowej)
3. [Przestrzeń stanów](#3-przestrzeń-stanów)
4. [Przesiadki i funkcja kosztu](#4-przesiadki-i-funkcja-kosztu)
5. [Konstrukcja MCMC](#5-konstrukcja-mcmc)
6. [Propozycje ruchów strukturalnych](#6-propozycje-ruchów-strukturalnych)
7. [Inercyjny Parallel Tempering](#7-inercyjny-parallel-tempering)
8. [Analiza zbieżności i diagnostyka](#8-analiza-zbieżności-i-diagnostyka)
9. [Metryki jakości rozwiązania](#9-metryki-jakości-rozwiązania)
10. [Złożoność obliczeniowa](#10-złożoność-obliczeniowa)
11. [Kwestie do zbadania empirycznie](#11-kwestie-do-zbadania-empirycznie)

---

## 1. Wprowadzenie i motywacja

### 1.1 Problem PESP

Problem PESP (Periodic Event Scheduling Problem) został wprowadzony przez Serafiniego i Ukovicha (1989) jako uogólniony model synchronizacji zdarzeń okresowych. W kontekście transportu publicznego, PESP modeluje problem konstrukcji rozkładu jazdy minimalizującego czasy przesiadek przy zachowaniu okresowości.

**Definicja (Klasyczny PESP):**
Dany jest skierowany graf $G = (V, E)$ gdzie $V$ to zbiór zdarzeń (odjazdy i przyjazdy), $E$ to zbiór ograniczeń czasowych. Dla każdej krawędzi $(i,j) \in E$ dane są granice $[l_{ij}, u_{ij}]$ oraz okres $T > 0$. Problem polega na znalezieniu przypisania czasów $\pi: V \to \mathbb{Z}/T\mathbb{Z}$ takiego, że:

$$l_{ij} \leq (\pi_j - \pi_i) \bmod T \leq u_{ij} \quad \forall (i,j) \in E$$

Problem PESP jest NP-trudny.

### 1.2 Podejście MCMC z miękkimi ograniczeniami

Tradycyjne podejście MCMC traktujące ograniczenia jako nieprzekraczalne bariery ($\Cost = +\infty$) prowadzi do rozspójnienia przestrzeni stanów i braku ergodyczności. Przyjmujemy podejście **relaksacji ograniczeń**: każde ograniczenie twarde $g(x) \leq 0$ zastępujemy skończoną funkcją kary $\Psi(g(x))$, co zapewnia spójność przestrzeni stanów i ergodyczność łańcucha.

### 1.3 Status metodologiczny algorytmu

Algorytm jest specyfikowany w dwóch trybach:

- **Tryb A — Sampling (formalny MCMC):** Stały $\beta$, adaptacja z warunkiem *diminishing* (Sekcja 6.6), poprawna stacjonarność. Daje rozkład rozwiązań, pozwala na analizę niepewności.

- **Tryb B — Optimization (metaheurystyka):** Rosnący $\beta$ (schemat chłodzenia), ciągła adaptacja bez wymogu *diminishing*. Output = $\arg\min$ po trajektorii (best-so-far). Heurystyka optymalizacyjna inspirowana MCMC, bez gwarancji stacjonarności.

Struktura propozycji, inercja i funkcja kosztu są identyczne w obu trybach. Różnica dotyczy reżimu $\beta$ i adaptacji.

---

## 2. Model sieci transportowej

### 2.1 Struktura hierarchiczna

**Definicja (Modalności):**
Niech $\mathcal{M} = \{m_1, m_2, \ldots, m_K\}$ będzie skończonym zbiorem środków transportu (modalności), uporządkowanym hierarchicznie:

$$m_1 \succ m_2 \succ \cdots \succ m_K$$

**Przykład:** $m_1$ = pociągi dalekobieżne (IC, TLK, EIC), $m_2$ = pociągi regionalne, $m_3$ = tramwaje, $m_4$ = autobusy miejskie.

### 2.2 Lokalizacje i stacje

**Definicja (Lokalizacja i stacja):**
*Lokalizacja* $\ell \in \mathcal{P}$ to fizyczne miejsce. *Stacja* to para $(m, \ell) \in \mathcal{M} \times \mathcal{P}$ reprezentująca punkt obsługi modalności $m$ w lokalizacji $\ell$.

$$\mathcal{S} = \{(m, \ell) : m \in \mathcal{M}, \ell \in \mathcal{P}, \text{ modalność } m \text{ obsługuje } \ell\}$$

### 2.3 Linie i kursy

**Definicja (Linia):**
*Linia* $L$ jest krotką:

$$L = (m_L, \mathbf{s}_L, \boldsymbol{\delta}_L, \boldsymbol{\sigma}_L, f_L)$$

gdzie:
- $m_L \in \mathcal{M}$ — modalność linii
- $\mathbf{s}_L = (s_1, s_2, \ldots, s_{n_L}) \in \mathcal{S}^{n_L}$ — uporządkowana sekwencja stacji
- $\boldsymbol{\delta}_L = (\delta_1, \ldots, \delta_{n_L-1}) \in \mathbb{R}_{>0}^{n_L-1}$ — czasy przejazdów między stacjami
- $\boldsymbol{\sigma}_L = (\sigma_1, \ldots, \sigma_{n_L}) \in \mathbb{R}_{\geq 0}^{n_L}$ — minimalne czasy postoju
- $f_L \in \mathbb{Z}_{>0}$ — częstotliwość w minutach (np. $f_L = 60$)

**Założenie (Jednolity okres — PoC):**
Dla celów Proof of Concept przyjmujemy wspólny okres $T = 1440$ (minuty w dobie) dla wszystkich linii. Linia $L$ o częstotliwości $f_L$ ma:

$$N_L = T / f_L$$

kursów na okres. Czasy odjazdów z pierwszej stacji: $\theta_L + k \cdot f_L$ dla $k = 0, 1, \ldots, N_L - 1$.

Linie kursują ze stałą częstotliwością przez cały okres $T$.

**Uwaga (Rozszerzenie przyszłe):** Modelowanie linii nocnych i sklejanie rozkładów dzień/noc planowane jest po walidacji PoC. Uogólnienie na różne $T_L$ wymaga przeformułowania przesiadek w horyzoncie $\text{lcm}(T_{L_1}, T_{L_2})$.

**Definicja (Kurs):**
Dla linii $L$, *kurs* $\kappa = (L, k)$ z $k \in \{0, 1, \ldots, N_L - 1\}$ jest $k$-tym uruchomieniem linii w okresie $[0, T)$.

$$\mathcal{K} = \bigcup_{L \in \mathcal{L}} \{(L, k) : k = 0, \ldots, N_L - 1\}$$

### 2.4 Zdarzenia

Czas przyjazdu i odjazdu kursu $\kappa = (L, k)$ na stacji $s_i$:

$$t^{\text{arr}}(\kappa, s_i) = \theta_L + k \cdot f_L + \sum_{j=1}^{i-1} \delta_j + \sum_{j=1}^{i-1} \sigma_j$$

$$t^{\text{dep}}(\kappa, s_i) = t^{\text{arr}}(\kappa, s_i) + \sigma_i$$

Czasy rozumiane modulo $T$.

### 2.5 Ograniczenie obiegowe (Turnaround)

**Definicja (Turnaround):**
Dla pary łuków obrotu $(u, v) \in A_{\text{turn}}$, gdzie $u$ to przyjazd końcowy, a $v$ to odjazd początkowy:

$$w_{\text{turn}}(u, v) = (\pi_v - \pi_u) \bmod T$$

Wymagamy $w_{\text{turn}} \geq t_{\text{min\_turn}}$ (czas technologiczny).

---

## 3. Przestrzeń stanów

### 3.1 Reprezentacja rozkładu

**Definicja (Przestrzeń stanów):**
*Stan* (rozkład jazdy) to wektor offsetów wszystkich linii:

$$x = (\theta_L)_{L \in \mathcal{L}} \in \mathcal{X} = [0, T)^{|\mathcal{L}|}$$

**Każdy punkt** $x \in \mathcal{X}$ jest dopuszczalny w sensie MCMC, choć może mieć bardzo wysoki koszt. Gwarantuje to spójność przestrzeni.

### 3.2 Struktura torusowa

$$\mathcal{X} \cong \mathbb{T}^{|\mathcal{L}|} = (S^1)^{|\mathcal{L}|} \quad \text{gdzie } S^1 = \mathbb{R} / T\mathbb{Z}$$

Przestrzeń jest zwarta, spójna i nie ma brzegu — korzystne własności dla MCMC.

### 3.3 Dyskretyzacja

$$\mathcal{X}_{\text{disc}} = \{0, 1, \ldots, T - 1\}^{|\mathcal{L}|}$$

Offsety wyrażone w minutach. Dla $T = 1440$: $|\mathcal{X}_{\text{disc}}| = 1440^{|\mathcal{L}|}$.

---

## 4. Przesiadki i funkcja kosztu

### 4.1 Model przesiadek

**Definicja (Przesiadka — między liniami):**
*Przesiadka* $\gamma$ jest krotką:

$$\gamma = (L_1, s_1, L_2, s_2, \rho)$$

gdzie:
- $L_1$ — linia, z której pasażer wysiada
- $s_1$ — stacja wysiadania
- $L_2$ — linia, na którą pasażer wsiada
- $s_2$ — stacja wsiadania
- $\rho \geq 0$ — czas przejścia między $s_1$ a $s_2$

Przesiadka jest relacją między **liniami**, nie między konkretnymi kursami.

**Definicja (Czas oczekiwania):**
Dla przesiadki $\gamma = (L_1, s_1, L_2, s_2, \rho)$ i stanu $x$:

$$w_\gamma(x) = \bigl(-\Delta_\gamma(x)\bigr) \bmod f_{L_2}$$

gdzie $\Delta_\gamma(x) = a_{L_1, s_1}(x) + \rho - d^{(0)}_{L_2, s_2}(x)$ jest różnicą faz, $a_{L_1, s_1}(x)$ to czas przyjazdu $L_1$ na $s_1$, a $d^{(0)}_{L_2, s_2}(x)$ to czas pierwszego odjazdu $L_2$ z $s_2$.

**Własności czasu oczekiwania:**
1. $w_\gamma(x) \in [0, f_{L_2})$ — **zawsze nieujemny** z definicji
2. $\Delta_\gamma(x)$ jest **afiniczny** w $\theta_{L_1}$ i $\theta_{L_2}$: $\Delta_\gamma(x) = \theta_{L_1} - \theta_{L_2} + c_\gamma$ dla stałej $c_\gamma$ zależnej od czasów przejazdów, postojów i $\rho$
3. $w_\gamma(x)$ jest **kawałkami liniowa** z jednym skokiem (z $0$ do $f_{L_2}$) na okres $f_{L_2}$ — ciągła prawie wszędzie
4. Przy stałej częstotliwości $f_{L_2}$, czas oczekiwania jest **taki sam** dla każdego kursu danej linii — definicja „między liniami" nie traci informacji

### 4.2 Funkcje barier (Penalty Functions)

**Definicja (Bariera kwadratowa):**
Dla ograniczenia $w \geq w_{\min}$:

$$\Psi(w; w_{\min}, M) = M \cdot \left(\max(0, w_{\min} - w)\right)^2$$

gdzie $M \gg 0$ jest współczynnikiem kary ($M < \infty$).

### 4.3 Kalibracja parametru M

**Definicja (Kalibracja M):**
Niech $\epsilon_{\text{tol}}$ będzie akceptowalnym naruszeniem (np. 0.5 minuty), przy którym koszt bariery równa się typowemu kosztowi „złej" przesiadki $\Cost_{\text{max\_pax}}$:

$$M = \frac{\Cost_{\text{max\_pax}}}{(\epsilon_{\text{tol}})^2}$$

**Procedura kalibracji:** $M_{\text{initial}} = 1000 \cdot \bar{c}_{\text{ideal}}$. Po 10 000 krokach MCMC: $M_{\text{final}} = 100 \cdot \bar{C}_{\text{observed}}$.

### 4.4 Całkowita funkcja kosztu

$$\Cost(x) = \Cost_{\text{PAX}}(x) + \Cost_{\text{OPS}}(x) + \Cost_{\text{ROB}}(x)$$

#### 4.4.1 Koszt pasażerski

$$\Cost_{\text{PAX}}(x) = \sum_{\gamma \in \Gamma} \omega_\gamma \cdot c_{\text{soft}}(w_\gamma(x))$$

**Definicja (Miękki koszt przesiadki):**

$$c_{\text{soft}}(w) = \begin{cases}
\Psi(w; w_{\min}^\tau, M_{\text{safety}}) & \text{gdy } w < w_{\min}^\tau \\
\alpha_\tau (w_{\text{ideal}}^\tau - w)^2 & \text{gdy } w_{\min}^\tau \leq w < w_{\text{ideal}}^\tau \\
\beta_\tau (w - w_{\text{ideal}}^\tau)^2 & \text{gdy } w \geq w_{\text{ideal}}^\tau
\end{cases}$$

**Parametry typowe:**

| Typ przesiadki | $w_{\min}$ | $w_{\text{ideal}}$ | $w_{\max}$ | $\alpha$ | $\beta$ |
|---|---|---|---|---|---|
| IC → IC | 5 | 12 | 30 | 3.0 | 0.5 |
| IC → Regionalny | 8 | 15 | 45 | 2.5 | 0.4 |
| Regionalny → Autobus | 5 | 10 | 30 | 2.0 | 0.5 |
| Tramwaj → Tramwaj | 2 | 5 | 15 | 2.0 | 0.8 |

#### 4.4.2 Koszt operacyjny (obiegi i flota)

$$\Cost_{\text{OPS}}(x) = \Cost_{\text{turn}}(x) + \Cost_{\text{fleet}}(x)$$

**Obroty taboru:**

$$\Cost_{\text{turn}}(x) = \sum_{a \in A_{\text{turn}}} \Psi(w_a(x); t_{\text{min\_turn}}^a, M_{\text{critical}})$$

gdzie $M_{\text{critical}} > M_{\text{safety}}$.

**Proxy intensywności floty (Little's Law):**

$$\Cost_{\text{fleet}}(x) = \lambda_{\text{fleet}} \cdot \max(0, N_{\text{fleet}}(x) - N_{\text{available}})^2$$

gdzie:

$$N_{\text{fleet}}(x) \approx \frac{\sum_{e \in A_{\text{drive}}} \text{duration}(e) + \sum_{v \in V} \text{dwell}(v) + \sum_{a \in A_{\text{turn}}} w_a(x)}{T}$$

**Uwaga:** $N_{\text{fleet}}(x)$ jest **przybliżeniem średniej intensywności użycia taboru** (Little's Law), nie dolnym ograniczeniem wynikającym z maksymalnej jednoczesnej liczby pojazdów. Poprawne dolne ograniczenie — $N^{\text{peak}}_{\text{conc}}(x)$ — jest raportowane jako metryka diagnostyczna (Sekcja 9), ale nie wchodzi do $\Cost$.

#### 4.4.3 Proxy robustności

**Definicja (Kara za brak robustności):**
Dla przesiadki $\gamma$ z buforem $b_\gamma(x) = w_\gamma(x) - \rho_\gamma$ i parametrem opóźnień $\lambda_\gamma$ (zakładając $\Delta \sim \text{Exp}(\lambda_\gamma)$):

$$\Cost_{\text{ROB}}(\gamma, x) = \omega_\gamma \cdot \max\left(0, \frac{-\ln(1-\alpha) + 1}{\lambda_\gamma} - b_\gamma(x)\right)$$

Karamy przesiadki, których bufor czasowy nie pokrywa $\text{CVaR}_\alpha$ rozkładu opóźnień. Formuła jest **heurystycznym proxy robustności**, nie formalną minimalizacją $\text{CVaR}_\alpha$ straty pasażera.

$$\Cost_{\text{ROB}}(x) = \sum_{\gamma \in \Gamma} \Cost_{\text{ROB}}(\gamma, x)$$

Złożoność: $O(1)$ per przesiadka.

### 4.5 Wagi przesiadek

**Definicja (Waga przesiadki):**

$$\omega_\gamma = \omega_\gamma^{\text{pax}} \cdot \omega_\gamma^{\text{uniq}} \cdot \omega_\gamma^{\text{peak}} \cdot \omega_\gamma^{\text{hier}}$$

gdzie $\omega^{\text{pax}}_\gamma$ — szacowana liczba pasażerów, $\omega^{\text{uniq}}_\gamma \in \{1, \mu\}$ z $\mu > 1$ dla unikalnych połączeń, $\omega^{\text{peak}}_\gamma \in \{1, \nu\}$ z $\nu > 1$ dla godzin szczytu, $\omega^{\text{hier}}_\gamma$ — priorytet hierarchiczny.

---

## 5. Konstrukcja MCMC

### 5.1 Rozkład docelowy

**Definicja (Rozkład Boltzmanna):**
Dla parametru temperatury odwrotnej $\beta > 0$:

$$\pi_\beta(x) = \frac{1}{Z_\beta} \exp(-\beta \cdot \Cost(x))$$

gdzie $Z_\beta = \int_{\mathcal{X}} \exp(-\beta \cdot \Cost(x)) \, dx$.

Dla $\beta \to \infty$: $\pi_\beta \to \text{Uniform}(\arg\min_{x \in \mathcal{X}} \Cost(x))$.

### 5.2 Kernel przejścia Metropolis-Hastings

```
Input: Stan obecny x, temperatura odwrotna β, kernel propozycji q
Output: Nowy stan x'

1. Wylosuj kandydata y ~ q(·|x)
2. Oblicz współczynnik akceptacji:
   α(x, y) = min(1, exp(-β(Cost(y) - Cost(x))) · q(x|y)/q(y|x))
3. Wylosuj u ~ Uniform(0, 1)
4. If u < α(x, y): Return y  (akceptacja)
   Else: Return x  (odrzucenie)
```

Dla symetrycznych propozycji ($q(y|x) = q(x|y)$):
$\alpha(x, y) = \min(1, e^{-\beta(\Cost(y) - \Cost(x))})$

### 5.3 Własności ergodyczne

**Twierdzenie (Ergodyczność):**
Łańcuch Markowa z kernelem MH jest ergodyczny z rozkładem stacjonarnym $\pi_\beta$ jeśli:

1. $\Cost(x) < \infty$ dla wszystkich $x \in \mathcal{X}$ (gwarantowane przez miękkie ograniczenia) — zapewnia $\pi_\beta(x) > 0$ wszędzie
2. Kernel propozycji $q$ spełnia:
   - (i) Każda linia $L \in \mathcal{L}$ ma dodatnie prawdopodobieństwo wyboru: $P(L) > 0$
   - (ii) Rozkład przesunięcia $\delta$ ma wsparcie całe $\mathbb{Z}$ (przypadek dyskretny) lub pełne wsparcie na $\mathbb{R}$ (przypadek ciągły)
3. $\Cost$ nie jest stałe (gwarantuje aperiodyczność przez dodatnie prawdopodobieństwo odrzucenia)

**Dowód (szkic):**
- Szczegółowa równowaga z konstrukcji MH: $\pi_\beta(x) \cdot P(x,y) = \pi_\beta(y) \cdot P(y,x)$
- Warunek 1: akceptacja $\alpha(x,y) > 0$ dla każdej pary $(x,y)$ z $q(y|x) > 0$ — target nigdzie nie „zabija" przejść
- Warunek 2(i) + 2(ii): z dowolnego $x$ kernel $q$ daje dodatnie prawdopodobieństwo przejścia do dowolnego $y$ w co najwyżej $|\mathcal{L}|$ krokach. W przypadku dyskretnym: nieobcięty Gauss z zaokrągleniem daje wsparcie na całym $\mathbb{Z}$, co generuje grupę $\mathbb{Z}_T$
- Warunek 3: $\Cost$ niestałe ⟹ ∃ $x,y$: $\alpha(x,y) < 1$ ⟹ dodatnie prawdopodobieństwo pozostania w tym samym stanie ⟹ okres = 1

**Uwaga:** Warunek 2(i) jest automatycznie spełniony przez inercję: $P(L) \propto 1/I_L$ z $I_L \geq 1 < \infty$ daje $P(L) > 0$.

**Uwaga:** Dodatnia gęstość $\pi_\beta$ (Warunek 1) jest warunkiem **koniecznym**, ale **niewystarczającym** dla nieprzywiedlności. Nieprzywiedlność zależy wyłącznie od kernela $q$ (Warunek 2), nie od samej funkcji energii.

---

## 6. Propozycje ruchów strukturalnych

### 6.1 Przesunięcie pojedynczej linii (Single Line Shift)

**Definicja:**
Dla linii $L$ i skali $\sigma > 0$:

$$q_{\text{single}}(x' | x): \quad \theta'_L = (\theta_L + \epsilon) \bmod T, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

Pozostałe współrzędne bez zmian. Kernel symetryczny.

### 6.2 Przesunięcie komponentu (Component Move)

**Algorytm (ComponentPropagator):**

```
Input: Linia startowa L_seed, parametr zaniku p_decay
Output: Zbiór linii do przesunięcia C

1. Queue = [L_seed], Visited = {L_seed}
2. While Queue niepusta:
     a. Current = Queue.pop()
     b. For each sąsiad Next w G_dep:
          If Next not in Visited:
            r ~ Uniform(0, 1)
            If r > p_decay:
              Visited.add(Next), Queue.append(Next)
3. Return Visited
```

**Propozycja:** $\theta'_L = (\theta_L + \delta) \bmod T$ dla wszystkich $L \in C$, gdzie $\delta \sim \mathcal{N}(0, \sigma^2)$.

**Dobór parametru zaniku:**

$$p_{\text{decay}} = \min\left(0.5, \frac{0.8}{d_{\text{avg}}}\right)$$

gdzie $d_{\text{avg}}$ to średni stopień w grafie zależności $G_{\text{dep}}$. Dla $d \cdot p_{\text{decay}} < 1$ oczekiwany rozmiar komponentu jest skończony.

### 6.3 Ruch naprawczy (Smart Repair Move)

**Algorytm (Smart Repair):**

```
Input: Stan x, zbiór przesiadek Γ
Output: Stan x', Hastings ratio H

1. Wybierz przesiadkę γ ~ Uniform(Γ)
2. Niech L_1, L_2 = linie zaangażowane w γ
3. Wybierz linię L ~ Uniform({L_1, L_2})
4. If w_γ(x) ∈ [w_min, w_max]:
     δ ~ N(0, σ_small²)
   Else:
     Oblicz δ* = argmin_{d ∈ ℤ, w_γ(x ⊕_L d) = w_ideal} |d|  // reguła minimalnej normy
     δ = δ* + ε, gdzie ε ~ N(0, σ_small²)
5. θ'_L = (θ_L + δ) mod T
6. H = φ(δ - δ*_forward; 0, σ_small²) / φ(δ' - δ*_backward; 0, σ_small²)
7. Return x', H
```

**Uwaga:** Przy tie-breaku (dwa rozwiązania o tej samej odległości na okręgu) — losowy wybór z $p = 0.5$. Symetryczny, nie wpływa na Hastings ratio.

### 6.4 Swap offsetów

**Definicja:**
Dla dwóch linii $L_1, L_2$ tej samej modalności:

$$q_{\text{swap}}(x' | x): \quad \theta'_{L_1} = \theta_{L_2}, \quad \theta'_{L_2} = \theta_{L_1}$$

Kernel symetryczny (inwolucja).

### 6.5 Mieszanka propozycji

$$q(x' | x) = \sum_{j=1}^{J} p_j \cdot q_j(x' | x), \quad \sum_{j=1}^{J} p_j = 1$$

**Sugerowane wagi początkowe:**

| Propozycja | Waga $p_j$ |
|---|---|
| Single Line Shift | 0.4 |
| Component Move | 0.3 |
| Smart Repair | 0.2 |
| Swap | 0.1 |

### 6.6 Adaptacja wag propozycji

**Definicja (Adaptacja — Tryb A / Sampling):**
Aktualizacja wag:

$$p_j^{(n+1)} = \Pi_\Delta\!\left(p_j^{(n)} + \gamma_n \cdot g(\hat{r}_j(n))\right)$$

gdzie $\Pi_\Delta$ to projekcja na sympleks, $g(\hat{r}_j) = \hat{r}_j(1 - \hat{r}_j) - \bar{g}$ preferuje umiarkowany acceptance rate, $\gamma_n = n^{-a}$ z $a \in (1/2, 1]$ (domyślnie $a = 0.7$).

Warunki gwarantujące poprawną stacjonarność:
1. **Diminishing adaptation**: $\gamma_n \to 0$, $\sum_n \gamma_n = \infty$ ⟹ $\|p^{(n+1)} - p^{(n)}\| \leq C\gamma_n \to 0$
2. **Containment**: $p_j(n) \geq \varepsilon > 0$ dla każdego $j$, $\sigma \in [\sigma_{\min}, \sigma_{\max}]$

Opcjonalnie: zamrożenie adaptacji po $N_0$ krokach.

**Definicja (Adaptacja — Tryb B / Optimization):**
Ciągła adaptacja bez wymogu *diminishing*. Algorytm nie jest formalnie poprawnym MCMC. Output: $\arg\min$ po trajektorii (best-so-far).

---

## 7. Inercyjny Parallel Tempering

### 7.1 Parallel Tempering — podstawy

Rozważamy $M$ replik łańcucha z temperaturami $\beta_1 < \beta_2 < \cdots < \beta_M$.

Stan rozszerzony: $\mathbf{x} = (x^{(1)}, \ldots, x^{(M)}) \in \mathcal{X}^M$

Rozkład docelowy: $\Pi(\mathbf{x}) = \prod_{i=1}^{M} \pi_{\beta_i}(x^{(i)})$

**Schemat temperatur (geometryczny):**

$$\beta_i = \beta_1 \cdot r^{i-1}, \quad r = \left(\frac{\beta_M}{\beta_1}\right)^{1/(M-1)}$$

### 7.2 Mechanizm inercji

**Definicja (Inercja):**
Każda replika $i$ posiada wektor inercji $I^{(i)} \in \mathbb{R}^{|\mathcal{L}|}_{\geq 1}$. Prawdopodobieństwo wyboru linii $L$ do modyfikacji w replice $i$:

$$P^{(i)}(L) \propto \frac{1}{I^{(i)}_L}$$

**Parametry inercji:**

| Poziom modalności | $I_L$ |
|---|---|
| IC (dalekobieżny) | 20 |
| Regionalny | 5 |
| Lokalny | 1 |

**Konfiguracja replik:**
- Repliki „gorące" ($i \approx M$): $I^{(i)}_L = 1$ dla wszystkich $L$ (pełna swoboda)
- Repliki „zimne" ($i \approx 1$): $I^{(i)}_L$ według tabeli (IC stabilne, lokalne elastyczne)

### 7.3 Algorytm

```
Algorithm: MCMC-PESP with Inertial Parallel Tempering

Input:
  - Network: L (lines), Γ (transfers), A_turn (turnarounds)
  - Parameters: M (replicas), β_min, β_max, I (inertia vector)
  - Stopping: max_iterations, convergence_threshold

Output:
  - Best schedule x*

Initialize:
  1. β_i = β_min * (β_max/β_min)^((i-1)/(M-1)) for i = 1..M
  2. x^(i) = random_schedule() for i = 1..M
  3. x* = argmin_{x^(i)} Cost(x^(i))
  4. Calibrate M_barrier via calibration_run()

Main Loop:
  For iter = 1 to max_iterations:
    
    // Local moves (parallel)
    For i = 1 to M:
      proposal_type = sample_proposal(weights)
      L = sample_line(P ∝ 1/I^(i)_L)
      Execute proposal, compute ΔCost incrementally
      Accept/reject via MH
      If Cost(x') < Cost(x*): x* = x'
    
    // Replica exchange
    i = random(1, M-1)
    α_swap = min(1, exp[(β_i - β_{i+1})(Cost(x^(i+1)) - Cost(x^(i)))])
    If Uniform(0,1) < α_swap: Swap x^(i) ↔ x^(i+1)
    
    // Diagnostics (every 1000 iterations)
    Compute R_hat, update proposal weights (Section 6.6)

Return x*
```

### 7.4 Poprawność

**Twierdzenie:** Inercyjny PT zachowuje poprawny rozkład stacjonarny.

**Dowód:** $I_L$ jest stałe (nie zależy od $x$), więc prawdopodobieństwo wyboru linii $L$ jest takie samo dla przejścia $x \to x'$ jak dla $x' \to x$. Kernel pozostaje symetryczny (lub z obliczalnym Hastings ratio). Każda replika $i$ próbkuje z $\pi_{\beta_i}$. Wymiana replik spełnia szczegółową równowagę dla rozkładu produktowego $\Pi$.

---

## 8. Analiza zbieżności i diagnostyka

### 8.1 Statystyka Gelman-Rubin

Dla $M$ niezależnych łańcuchów o długości $N$:

$$\hat{R} = \sqrt{\frac{\hat{V}}{W}}$$

gdzie $W = \frac{1}{M} \sum_{m=1}^{M} s_m^2$ (wariancja wewnątrz łańcuchów), $B = \frac{N}{M-1} \sum_{m=1}^{M} (\bar{x}_m - \bar{x})^2$ (wariancja między łańcuchami), $\hat{V} = \frac{N-1}{N} W + \frac{1}{N} B$.

**Kryterium zbieżności:** $\hat{R} < 1.1$

### 8.2 Efektywna wielkość próby

$$\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^{\infty} \rho_k}$$

**Kryterium:** $\text{ESS} > 100$

### 8.3 Tempo zbieżności

**Twierdzenie (Tempo geometryczne):**
Jeśli łańcuch jest geometrycznie ergodyczny: $\|P^n(x, \cdot) - \pi_\beta\|_{TV} \leq C(x) \cdot \rho^n$ dla $C(x) < \infty$, $\rho < 1$.

---

## 9. Metryki jakości rozwiązania

### 9.1 Metryki operacyjne

| Metryka | Definicja | Cel |
|---|---|---|
| $\bar{w}(x)$ | Średni ważony czas przesiadki | Minimalizuj |
| MissRate$(x)$ | % przesiadek z $w < w_{\min}$ | = 0 |
| LongRate$(x)$ | % przesiadek z $w > w_{\max}$ | Minimalizuj |
| $\sigma_w(x)$ | Odchylenie standardowe czasów | Minimalizuj |
| $w_{\max}(x)$ | Najdłuższa przesiadka | Minimalizuj |

### 9.2 Metryki robustności

- BufferAdequacy$(x)$ — % przesiadek z buforem $\geq q_{0.9}$ opóźnienia
- CascadeRisk$(x)$ — max zasięg propagacji opóźnienia w grafie

### 9.3 Metryki diagnostyczne

**Definicja (Peak concurrency):**

$$N^{\text{peak}}_{\text{conc}}(x) = \max_{t \in [0,T)} \sum_{\kappa \in \mathcal{K}} \mathbf{1}\{t \in [t^{\text{dep}}_\kappa(x), t^{\text{arr}}_\kappa(x))\}$$

Obliczalna w $O(|\mathcal{K}| \log |\mathcal{K}|)$ algorytmem sweep-line. Raportowana jako diagnostyka (dolne ograniczenie na flotę), **nie wchodzi do $\Cost$**.

### 9.4 Metryki pasażerskie

- WeightedJourneyTime$(x)$ — średni ważony czas podróży end-to-end
- Accessibility$_T(x)$ — % par OD osiągalnych w czasie $\leq T$

---

## 10. Złożoność obliczeniowa

### 10.1 Koszt jednego kroku

| Operacja | Złożoność naiwna | Złożoność inkrementalna |
|---|---|---|
| Obliczenie $\Cost(x)$ | $O(|\Gamma|)$ | — |
| $\Delta\Cost$ (single line) | $O(|\Gamma|)$ | $O(|\Gamma_L|)$ |
| Component Move | $O(|C| \cdot |\Gamma|)$ | $O(|\Gamma_C|)$ |
| $N^{\text{peak}}_{\text{conc}}$ | — | $O(|\mathcal{K}| \log |\mathcal{K}|)$ |

### 10.2 Graf zależności

$G_{\text{dep}} = (\mathcal{L}, E_{\text{dep}})$ gdzie $(L_1, L_2) \in E_{\text{dep}}$ jeśli istnieje przesiadka między $L_1$ i $L_2$. Dla propozycji zmieniającej linie w zbiorze $S$:

$$\Gamma_{\text{affected}} = \{\gamma \in \Gamma : L_1(\gamma) \in S \text{ lub } L_2(\gamma) \in S\}$$

### 10.3 Równoległość

Dla PT z $M$ replikami na $M$ procesorach: lokalne kroki MH w pełni równoległe, wymiana $O(M)$ komunikacji. Speedup bliski $M$ dla dużych sieci.

---

## 11. Kwestie do zbadania empirycznie

Poniższe zagadnienia nie mogą być rozstrzygnięte teoretycznie i wymagają eksperymentów.

### 11.1 Mixing time i zbieżność

| Pytanie | Eksperyment | Metryka sukcesu |
|---|---|---|
| Ile kroków do zbieżności? | 4+ niezależne łańcuchy, różne punkty startowe | $\hat{R} < 1.1$ |
| Jaki ESS / krok? | Autokorelacja dla $\Cost(x)$ | ESS/krok > 0.01 |
| Czy łańcuch ucieka z minimów lokalnych? | Trace plot $\Cost(x)$ | Widoczne „skoki" |
| Jak mixing skaluje się z rozmiarem? | Sieci 10, 50, 100 linii | Mixing time $\propto |\mathcal{L}|^?$ |

**Minimalny eksperyment:** 10 linii, 20 przesiadek, 4 łańcuchy, 100k kroków.

### 11.2 Porównanie propozycji ruchów

| Pytanie | Eksperyment | Metryka |
|---|---|---|
| Która propozycja ma najlepszy ESS/krok? | MCMC z każdą propozycją osobno | ESS / czas CPU |
| Optymalny mix propozycji? | Grid search po wagach $p_j$ | Końcowa jakość + czas zbieżności |
| Czy Smart Repair przyspiesza zbieżność? | Porównaj z/bez Smart Repair | Czas do $\hat{R} < 1.1$ |
| Optymalny $p_{\text{decay}}$? | Testuj $p \in \{0.1, 0.3, 0.5, 0.7\}$ | ESS / krok |

### 11.3 Efektywność Parallel Tempering

| Pytanie | Eksperyment | Metryka |
|---|---|---|
| Ile replik optymalnie? | $M \in \{4, 8, 16, 32\}$ | Jakość / czas CPU |
| Jaki rozstaw temperatur? | Geometryczny vs adaptacyjny | Swap acceptance rate (cel: 20-30%) |
| Czy inercja pomaga? | Zwykłe PT vs inercyjne PT | Jakość końcowa + stabilność |

### 11.4 Kalibracja funkcji kosztu

| Pytanie | Eksperyment | Metryka |
|---|---|---|
| Wrażliwość na $M$ (bariera)? | $M \in \{10^3, 10^4, 10^5, 10^6\}$ | Stabilność wyników |
| Dobór $\lambda_{\text{fleet}}$, $\lambda_{\text{rob}}$? | Analiza Pareto | Trade-off krzywe |
| Czy proxy robustności daje lepsze rozwiązania niż brak? | Porównaj warianty | Robustność w symulacji |

### 11.5 Benchmark vs alternatywy

| Metoda | Eksperyment | Cel porównania |
|---|---|---|
| Random search | $N$ losowych stanów, weź najlepszy | Baseline minimum |
| Greedy local search | Iteracyjnie poprawiaj najgorsze ograniczenie | Czy MCMC bije heurystykę? |
| Simulated annealing | Te same propozycje, schedule chłodzenia | Czy formalizm MCMC daje przewagę? |
| MIP (małe instancje) | Rozwiąż do optymalności | Jak daleko od optimum? |
| Algorytm genetyczny | Standardowa implementacja | Porównanie z praktyką przemysłową |

### 11.6 Walidacja na rzeczywistych danych

Porównanie z GTFS rzeczywistej sieci, konsultacja z ekspertami, analiza wąskich gardeł.

### 11.7 Skalowanie

| Pytanie | Eksperyment | Metryka |
|---|---|---|
| Czas obliczeń vs $|\mathcal{L}|$? | Sieci 10, 50, 100, 500 linii | Czas do zbieżności |
| Czas vs $|\Gamma|$? | Stała liczba linii, różna gęstość | Czas / krok |
| Wąskie gardła obliczeniowe? | Profiling kodu | % czasu w każdej funkcji |
| Speedup inkrementalnego kosztu? | Porównaj z naiwną implementacją | Speedup factor |

---

## Podsumowanie

Model jest gotowy do implementacji. Zmiany względem wersji 1.0:

1. **Jednolity okres**: $T_L = T = 1440$ dla wszystkich linii. Linie o różnych częstotliwościach mają $N_L = T / f_L$ kursów.
2. **Przesiadki między liniami**: Nowa definicja — przesiadka jako relacja $(L_1, s_1, L_2, s_2, \rho)$ między liniami, nie kursami. Czas oczekiwania zawsze $\geq 0$.
3. **Ergodyczność — poprawiony dowód**: Explicite rozdziela warunki na target ($\Cost < \infty$) i kernel ($P(L) > 0$, wsparcie $\delta$ na całym $\mathbb{Z}$). Usunięta błędna implikacja.
4. **Robustność — uczciwe proxy**: Jawnie opisana jako heurystyczne proxy, nie formalna minimalizacja CVaR straty.
5. **Smart Repair — reguła minimalnej normy**: $\delta^*$ jako przesunięcie o minimalnej wartości bezwzględnej. Tie-break losowy.
6. **Adaptacja — dwa tryby**: Tryb A z diminishing adaptation i containment. Tryb B — ciągła adaptacja, jawnie nazwana metaheurystyką.
7. **Proxy floty — uczciwy opis**: $N_{\text{fleet}}$ jako średnia intensywność (Little's Law). $N^{\text{peak}}_{\text{conc}}$ jako diagnostyka.
8. **Status metodologiczny**: Jawny podział na Tryb A (sampling) i Tryb B (optimization).

---

## Appendix A: Notacja

| Symbol | Znaczenie |
|---|---|
| $\mathcal{M}$ | Zbiór modalności |
| $\mathcal{S}$ | Zbiór stacji |
| $\mathcal{L}$ | Zbiór linii |
| $\mathcal{K}$ | Zbiór kursów |
| $\Gamma$ | Zbiór przesiadek |
| $\mathcal{X}$ | Przestrzeń stanów |
| $x = (\theta_L)_L$ | Stan (wektor offsetów) |
| $T$ | Wspólny okres ($= 1440$ min) |
| $f_L$ | Częstotliwość linii $L$ |
| $N_L = T / f_L$ | Liczba kursów linii $L$ na okres |
| $w_\gamma(x)$ | Czas oczekiwania na przesiadkę $\gamma$ |
| $\Cost(x)$ | Całkowity koszt stanu $x$ |
| $\pi_\beta$ | Rozkład Boltzmanna z temp. odwrotną $\beta$ |
| $q(x'|x)$ | Kernel propozycji |
| $\alpha(x, y)$ | Prawdopodobieństwo akceptacji |
| $\Psi(\cdot)$ | Funkcja bariery (kary) |
| $G_{\text{dep}}$ | Graf zależności linii |
| $I_L$ | Inercja linii $L$ |
| $N^{\text{peak}}_{\text{conc}}$ | Peak concurrency (diagnostyka) |
