# Wprowadzenie i motywacja {#sec:intro}

## Problem PESP

Problem PESP (Periodic Event Scheduling Problem) został wprowadzony
przez Serafiniego i Ukovicha (1989) jako uogólniony model synchronizacji
zdarzeń okresowych.

::: {#def:pesp .definition}
**Definicja 1** (Klasyczny PESP). *Dany jest skierowany graf $G=(V,E)$,
gdzie $V$ to zbiór zdarzeń (odjazdy i przyjazdy), $E$ to zbiór
ograniczeń czasowych. Dla każdej krawędzi $(i,j)\in E$ dane są granice
$[l_{ij},u_{ij}]$ oraz okres $T>0$. Problem polega na znalezieniu
przypisania czasów $\pi\colon V\to\mathbb{Z}/T\mathbb{Z}$ takiego, że:
$$l_{ij} \;\leq\; (\pi_j - \pi_i) \bmod T \;\leq\; u_{ij}
  \qquad \forall\,(i,j)\in E$$ przy optymalizacji pewnej funkcji celu.
Problem PESP jest NP-trudny.*
:::

## Podejście MCMC z miękkimi ograniczeniami

Tradycyjne podejście traktujące ograniczenia jako nieprzekraczalne
bariery ($\mathrm{Cost}=+\infty$) prowadzi do rozspójnienia przestrzeni
stanów i braku ergodyczności.

W niniejszym dokumencie przyjmujemy podejście **relaksacji ograniczeń**:
każde ograniczenie twarde $g(x)\leq 0$ zastępujemy skończoną funkcją
kary $\Psi(g(x))$, co zapewnia spójność przestrzeni stanów
i ergodyczność łańcucha.

## Status metodologiczny algorytmu {#sec:status}

Algorytm jest specyfikowany w dwóch trybach:

::: description
Stały $\beta$, adaptacja z warunkiem *diminishing*
(Sekcja [6.6](#sec:adaptation){reference-type="ref"
reference="sec:adaptation"}), poprawna stacjonarność. Daje rozkład
rozwiązań, pozwala na analizę niepewności.

Rosnący $\beta$ (schemat chłodzenia), ciągła adaptacja bez wymogu
*diminishing*. Output $=\arg\min$ po trajektorii (best-so-far).
Heurystyka optymalizacyjna inspirowana MCMC, bez gwarancji
stacjonarności.
:::

Struktura propozycji, inercja i funkcja kosztu są identyczne w obu
trybach. Różnica dotyczy reżimu $\beta$ i adaptacji.

# Model sieci transportowej {#sec:network}

## Struktura hierarchiczna

::: definition
**Definicja 2** (Modalności). *Niech
$\mathcal{M}=\{m_1,m_2,\ldots,m_K\}$ będzie skończonym zbiorem środków
transportu (modalności), uporządkowanym hierarchicznie:
$m_1\succ m_2\succ\cdots\succ m_K$, gdzie $m_i\succ m_j$ oznacza, że
modalność $m_i$ ma wyższy priorytet.*
:::

**Przykład**: $m_1$ = pociągi dalekobieżne (IC, TLK, EIC), $m_2$ =
pociągi regionalne, $m_3$ = tramwaje, $m_4$ = autobusy miejskie.

## Lokalizacje i stacje

::: definition
**Definicja 3** (Lokalizacja i stacja). **Lokalizacja*
$\ell\in\mathcal{P}$ to fizyczne miejsce (dworzec, przystanek). *Stacja*
to para $(m,\ell)\in\mathcal{M}\times\mathcal{P}$ reprezentująca punkt
obsługi modalności $m$ w lokalizacji $\ell$.
$$\mathcal{S} = \{(m,\ell) : m\in\mathcal{M},\;\ell\in\mathcal{P},
    \;\text{modalność }m\text{ obsługuje }\ell\}$$*
:::

::: remark
**Uwaga 1**. *Rozdzielenie modalności w tej samej lokalizacji pozwala
modelować różne czasy przejścia między peronami, różne częstotliwości
itp.*
:::

## Linie i kursy

::: {#def:line .definition}
**Definicja 4** (Linia). **Linia* $L$ jest krotką:
$$L = (m_L,\;\mathbf{s}_L,\;\boldsymbol{\delta}_L,\;\boldsymbol{\sigma}_L,\;f_L)$$
gdzie:*

-   *$m_L\in\mathcal{M}$ --- modalność linii,*

-   *$\mathbf{s}_L=(s_1,\ldots,s_{n_L})\in\mathcal{S}^{n_L}$ ---
    uporządkowana sekwencja stacji,*

-   *$\boldsymbol{\delta}_L=(\delta_1,\ldots,\delta_{n_L-1})\in\mathbb{R}_{>0}^{n_L-1}$
    --- czasy przejazdów,*

-   *$\boldsymbol{\sigma}_L=(\sigma_1,\ldots,\sigma_{n_L})\in\mathbb{R}_{\geq 0}^{n_L}$
    --- minimalne czasy postoju,*

-   *$f_L\in\mathbb{Z}_{>0}$ --- częstotliwość w minutach
    (np. $f_L=60$).*
:::

::: {#asm:uniform_period .assumption}
**Założenie 1** (Jednolity okres --- PoC). *Dla celów Proof of Concept
przyjmujemy wspólny okres $T=1440$ (minuty w dobie) dla wszystkich
linii. Linia $L$ o częstotliwości $f_L$ ma $$N_L = T / f_L$$ kursów na
okres. Czasy odjazdów z pierwszej stacji: $\theta_L + k\cdot f_L$ dla
$k=0,1,\ldots,N_L-1$.*

*Linie kursują ze stałą częstotliwością przez cały okres $T$.*
:::

::: remark
**Uwaga 2** (Rozszerzenie przyszłe). *Modelowanie linii nocnych
i sklejanie rozkładów dzień/noc planowane jest po walidacji PoC.
Uogólnienie na różne $T_L$ wymaga przeformułowania przesiadek
w horyzoncie $\mathrm{lcm}(T_{L_1},T_{L_2})$.*
:::

::: definition
**Definicja 5** (Kurs). *Dla linii $L$, *kurs* $\kappa=(L,k)$
z $k\in\{0,1,\ldots,N_L-1\}$ jest $k$-tym uruchomieniem linii w okresie
$[0,T)$.
$$\mathcal{K} = \bigcup_{L\in\mathcal{L}}\{(L,k):k=0,\ldots,N_L-1\}$$*
:::

## Zdarzenia

Czas przyjazdu i odjazdu kursu $\kappa=(L,k)$ na stacji $s_i$:
$$\begin{aligned}
  t^{\mathrm{arr}}(\kappa,s_i) &= \theta_L + k\cdot f_L
    + \sum_{j=1}^{i-1}\delta_j + \sum_{j=1}^{i-1}\sigma_j \\
  t^{\mathrm{dep}}(\kappa,s_i) &= t^{\mathrm{arr}}(\kappa,s_i) + \sigma_i
\end{aligned}$$ Czasy rozumiane modulo $T$.

## Ograniczenie obiegowe (Turnaround)

::: definition
**Definicja 6** (Turnaround). *Dla pary łuków obrotu
$(u,v)\in A_{\mathrm{turn}}$, gdzie $u$ to przyjazd końcowy linii, a $v$
to odjazd początkowy, czas dostępny na obrót:
$$w_{\mathrm{turn}}(u,v) = (\pi_v - \pi_u) \bmod T$$ Wymagamy
$w_{\mathrm{turn}}\geq t_{\mathrm{min\_turn}}$ (czas technologiczny).*
:::

# Przestrzeń stanów {#sec:state_space}

## Reprezentacja rozkładu

::: definition
**Definicja 7** (Przestrzeń stanów). **Stan* (rozkład jazdy) to wektor
offsetów wszystkich linii:
$$x = (\theta_L)_{L\in\mathcal{L}} \in \mathcal{X} = [0,T)^{|\mathcal{L}|}$$
**Każdy punkt** $x\in\mathcal{X}$ jest dopuszczalny w sensie MCMC, choć
może mieć bardzo wysoki koszt. Gwarantuje to spójność przestrzeni.*
:::

## Struktura torusowa

$$\mathcal{X} \cong \mathbb{T}^{|\mathcal{L}|} = (S^1)^{|\mathcal{L}|}
  \qquad\text{gdzie } S^1 = \mathbb{R}/T\mathbb{Z}$$ Przestrzeń jest
zwarta, spójna i nie ma brzegu --- korzystne własności dla MCMC.

## Dyskretyzacja

$$\mathcal{X}_{\mathrm{disc}} = \{0,1,\ldots,T-1\}^{|\mathcal{L}|}$$
Offsety wyrażone w minutach. Dla $T=1440$:
$|\mathcal{X}_{\mathrm{disc}}|=1440^{|\mathcal{L}|}$.

# Przesiadki i funkcja kosztu {#sec:cost}

## Model przesiadek

::: {#def:transfer .definition}
**Definicja 8** (Przesiadka --- między liniami). **Przesiadka* $\gamma$
jest krotką: $$\gamma = (L_1,\;s_1,\;L_2,\;s_2,\;\rho)$$ gdzie:*

-   *$L_1$ --- linia, z której pasażer wysiada,*

-   *$s_1$ --- stacja wysiadania,*

-   *$L_2$ --- linia, na którą pasażer wsiada,*

-   *$s_2$ --- stacja wsiadania,*

-   *$\rho\geq 0$ --- czas przejścia między $s_1$ a $s_2$.*

*Przesiadka jest relacją między **liniami**, nie między konkretnymi
kursami.*
:::

::: {#def:waiting_time .definition}
**Definicja 9** (Czas oczekiwania). *Dla przesiadki
$\gamma=(L_1,s_1,L_2,s_2,\rho)$ i stanu $x$, czas oczekiwania wynosi:
$$w_\gamma(x) = \bigl(-\Delta_\gamma(x)\bigr) \bmod f_{L_2}
  \label{eq:waiting_time}$$ gdzie
$\Delta_\gamma(x) = a_{L_1,s_1}(x) + \rho - d^{(0)}_{L_2,s_2}(x)$ jest
różnicą faz, $a_{L_1,s_1}(x)$ to czas przyjazdu $L_1$ na $s_1$,
a $d^{(0)}_{L_2,s_2}(x)$ to czas pierwszego odjazdu $L_2$ z $s_2$.*
:::

::: remark
**Uwaga 3** (Własności czasu oczekiwania).

1.  *$w_\gamma(x)\in[0,f_{L_2})$ --- **zawsze nieujemny** z definicji.*

2.  *$\Delta_\gamma(x)$ jest **afiniczny** w $\theta_{L_1}$
    i $\theta_{L_2}$:
    $\Delta_\gamma(x) = \theta_{L_1} - \theta_{L_2} + c_\gamma$ dla
    stałej $c_\gamma$ zależnej od czasów przejazdów, postojów i $\rho$.*

3.  *$w_\gamma(x)$ jest **kawałkami liniowa** z jednym skokiem (z $0$ do
    $f_{L_2}$) na okres $f_{L_2}$ --- ciągła prawie wszędzie.*

4.  *Przy stałej częstotliwości $f_{L_2}$, czas oczekiwania jest **taki
    sam** dla każdego kursu danej linii. Definicja "między liniami" nie
    traci informacji.*
:::

## Funkcje barier

::: definition
**Definicja 10** (Bariera kwadratowa). *Dla ograniczenia
$w\geq w_{\min}$:
$$\Psi(w;\;w_{\min},\;M) = M\cdot\bigl(\max(0,\;w_{\min}-w)\bigr)^2$$
gdzie $M\gg 0$ jest współczynnikiem kary ($M<\infty$).*
:::

::: definition
**Definicja 11** (Kalibracja $M$). *Niech $\epsilon_{\mathrm{tol}}$
będzie akceptowalnym naruszeniem (np. $0.5$ min), przy którym koszt
bariery równa się typowemu kosztowi "złej" przesiadki
$\mathrm{Cost}_{\mathrm{max\_pax}}$:
$$M = \frac{\mathrm{Cost}_{\mathrm{max\_pax}}}{\epsilon_{\mathrm{tol}}^2}$$
**Procedura kalibracji**:
$M_{\mathrm{initial}}=1000\cdot\bar{c}_{\mathrm{ideal}}$. Po $10\,000$
krokach MCMC: $M_{\mathrm{final}}=100\cdot\bar{C}_{\mathrm{observed}}$.*
:::

## Całkowita funkcja kosztu

$$\mathrm{Cost}(x) = \mathrm{Cost}_{\mathrm{PAX}}(x) + \mathrm{Cost}_{\mathrm{OPS}}(x) + \mathrm{Cost}_{\mathrm{ROB}}(x)
  \label{eq:total_cost}$$

### Koszt pasażerski

$$\mathrm{Cost}_{\mathrm{PAX}}(x) = \sum_{\gamma\in\Gamma}\omega_\gamma\cdot c_{\mathrm{soft}}(w_\gamma(x))$$

::: definition
**Definicja 12** (Miękki koszt przesiadki).
*$$c_{\mathrm{soft}}(w) = \begin{cases}
    \Psi(w;\;w_{\min}^\tau,\;M_{\mathrm{safety}})
      & \text{gdy } w < w_{\min}^\tau \\[3pt]
    \alpha_\tau\,(w_{\mathrm{ideal}}^\tau - w)^2
      & \text{gdy } w_{\min}^\tau\leq w < w_{\mathrm{ideal}}^\tau \\[3pt]
    \beta_\tau\,(w - w_{\mathrm{ideal}}^\tau)^2
      & \text{gdy } w\geq w_{\mathrm{ideal}}^\tau
  \end{cases}$$*
:::

::: {#tab:transfer_params}
  Typ przesiadki              $w_{\min}$   $w_{\mathrm{ideal}}$   $w_{\max}$   $\alpha$   $\beta$
  -------------------------- ------------ ---------------------- ------------ ---------- ---------
  IC $\to$ IC                     5                 12                30         3.0        0.5
  IC $\to$ Regionalny             8                 15                45         2.5        0.4
  Regionalny $\to$ Autobus        5                 10                30         2.0        0.5
  Tramwaj $\to$ Tramwaj           2                 5                 15         2.0        0.8

  : Parametry typowe dla różnych typów przesiadek
:::

### Koszt operacyjny

$$\mathrm{Cost}_{\mathrm{OPS}}(x) = \mathrm{Cost}_{\mathrm{turn}}(x) + \mathrm{Cost}_{\mathrm{fleet}}(x)$$

**Obroty taboru:**
$$\mathrm{Cost}_{\mathrm{turn}}(x) = \sum_{a\in A_{\mathrm{turn}}}
    \Psi\bigl(w_a(x);\;t^a_{\mathrm{min\_turn}},\;M_{\mathrm{critical}}\bigr)$$
gdzie $M_{\mathrm{critical}}>M_{\mathrm{safety}}$.

**Proxy intensywności floty (Little's Law):**
$$\mathrm{Cost}_{\mathrm{fleet}}(x) = \lambda_{\mathrm{fleet}}\cdot
    \max\bigl(0,\;N_{\mathrm{fleet}}(x)-N_{\mathrm{available}}\bigr)^2$$
gdzie: $$N_{\mathrm{fleet}}(x) \approx
    \frac{\displaystyle\sum_{e\in A_{\mathrm{drive}}}\mathrm{duration}(e)
    + \sum_{v\in V}\mathrm{dwell}(v)
    + \sum_{a\in A_{\mathrm{turn}}}w_a(x)}{T}$$

::: remark
**Uwaga 4**. *$N_{\mathrm{fleet}}(x)$ jest **przybliżeniem średniej
intensywności użycia taboru** (Little's Law), nie dolnym ograniczeniem
wynikającym z maksymalnej jednoczesnej liczby pojazdów. Ignoruje
strukturę obiegów i konflikty czasowe. Poprawne dolne ograniczenie ---
$N^{\mathrm{peak}}_{\mathrm{conc}}(x)$ --- jest raportowane jako metryka
diagnostyczna (Sekcja [9](#sec:metrics){reference-type="ref"
reference="sec:metrics"}), ale nie wchodzi do $\mathrm{Cost}$.*
:::

### Proxy robustności

::: {#def:robustness .definition}
**Definicja 13** (Kara za brak robustności). *Dla przesiadki $\gamma$
z buforem $b_\gamma(x)=w_\gamma(x)-\rho_\gamma$ i parametrem opóźnień
$\lambda_\gamma$ (zakładając $\Delta\sim\mathrm{Exp}(\lambda_\gamma)$):
$$\mathrm{Cost}_{\mathrm{ROB}}(\gamma,x) = \omega_\gamma\cdot
    \max\!\left(0,\;\frac{-\ln(1-\alpha)+1}{\lambda_\gamma} - b_\gamma(x)\right)
  \label{eq:robustness}$$ Karamy przesiadki, których bufor czasowy nie
pokrywa $\mathrm{CVaR}_\alpha$ rozkładu opóźnień. Formuła jest
**heurystycznym proxy robustności**, nie formalną minimalizacją
$\mathrm{CVaR}_\alpha$ straty pasażera.*
:::

$$\mathrm{Cost}_{\mathrm{ROB}}(x) = \sum_{\gamma\in\Gamma}\mathrm{Cost}_{\mathrm{ROB}}(\gamma,x)$$
Złożoność: $O(1)$ per przesiadka.

## Wagi przesiadek

::: definition
**Definicja 14** (Waga przesiadki).
*$$\omega_\gamma = \omega_\gamma^{\mathrm{pax}}\cdot\omega_\gamma^{\mathrm{uniq}}
    \cdot\omega_\gamma^{\mathrm{peak}}\cdot\omega_\gamma^{\mathrm{hier}}$$
gdzie $\omega^{\mathrm{pax}}_\gamma$ --- szacowana liczba pasażerów,
$\omega^{\mathrm{uniq}}_\gamma\in\{1,\mu\}$ z $\mu>1$ dla unikalnych
połączeń, $\omega^{\mathrm{peak}}_\gamma\in\{1,\nu\}$ z $\nu>1$ dla
godzin szczytu, $\omega^{\mathrm{hier}}_\gamma$ --- priorytet
hierarchiczny.*
:::

# Konstrukcja MCMC {#sec:mcmc}

## Rozkład docelowy

::: definition
**Definicja 15** (Rozkład Boltzmanna). *Dla parametru temperatury
odwrotnej $\beta>0$:
$$\pi_\beta(x) = \frac{1}{Z_\beta}\exp\bigl(-\beta\cdot\mathrm{Cost}(x)\bigr)$$
gdzie
$Z_\beta=\int_{\mathcal{X}}\exp(-\beta\cdot\mathrm{Cost}(x))\,dx$.*
:::

Dla $\beta\to\infty$:
$\pi_\beta\to\mathrm{Uniform}(\arg\min_{x\in\mathcal{X}}\mathrm{Cost}(x))$.

## Kernel przejścia Metropolis-Hastings

::: algorithm
Wylosuj kandydata $y\sim q(\cdot\mid x)$ Oblicz współczynnik akceptacji:
$\alpha(x,y)=\min\bigl(1,\;\exp(-\beta(\mathrm{Cost}(y)-\mathrm{Cost}(x)))\cdot q(x\mid y)/q(y\mid x)\bigr)$
Wylosuj $u\sim\mathrm{Uniform}(0,1)$
:::

Dla symetrycznych propozycji ($q(y\mid x)=q(x\mid y)$):
$\alpha(x,y) = \min(1,\;e^{-\beta(\mathrm{Cost}(y)-\mathrm{Cost}(x))})$.

## Własności ergodyczne

::: {#thm:ergodicity .theorem}
**Twierdzenie 1** (Ergodyczność). *Łańcuch Markowa z kernelem
Metropolis-Hastings jest ergodyczny z rozkładem stacjonarnym $\pi_\beta$
jeśli:*

1.  *$\mathrm{Cost}(x)<\infty$ dla wszystkich $x\in\mathcal{X}$
    (gwarantowane przez miękkie ograniczenia) --- zapewnia
    $\pi_\beta(x)>0$ wszędzie,*

2.  *Kernel propozycji $q$ spełnia:*

    1.  *Każda linia $L\in\mathcal{L}$ ma dodatnie prawdopodobieństwo
        wyboru: $P(L)>0$,*

    2.  *Rozkład przesunięcia $\delta$ ma wsparcie całe $\mathbb{Z}$
        (przypadek dyskretny) lub pełne wsparcie na $\mathbb{R}$
        (przypadek ciągły),*

3.  *$\mathrm{Cost}$ nie jest stałe (gwarantuje aperiodyczność przez
    dodatnie prawdopodobieństwo odrzucenia).*
:::

::: proof
*Dowód (szkic).*

-   Szczegółowa równowaga z konstrukcji MH:
    $\pi_\beta(x)\cdot P(x,y)=\pi_\beta(y)\cdot P(y,x)$.

-   Warunek 1 gwarantuje, że akceptacja $\alpha(x,y)>0$ dla każdej pary
    $(x,y)$ z $q(y\mid x)>0$ --- target nigdzie nie "zabija" przejść.

-   Warunek 2(i) + 2(ii): z dowolnego $x$ kernel $q$ daje dodatnie
    prawdopodobieństwo przejścia do dowolnego $y$ w co najwyżej
    $|\mathcal{L}|$ krokach (każdy ruszający inną linię). W przypadku
    dyskretnym: nieobcięty Gauss z zaokrągleniem daje wsparcie na całym
    $\mathbb{Z}$, co generuje grupę $\mathbb{Z}_T$.

-   Warunek 3: $\mathrm{Cost}$ niestałe $\Rightarrow$
    $\exists\,x,y: \alpha(x,y)<1$ $\Rightarrow$ dodatnie
    prawdopodobieństwo pozostania w tym samym stanie $\Rightarrow$ okres
    $=1$.

 ◻
:::

::: remark
**Uwaga 5**. *Warunek 2(i) jest automatycznie spełniony przez inercję
(Sekcja [7.2](#sec:inertia){reference-type="ref"
reference="sec:inertia"}): $P(L)\propto 1/I_L$ z $I_L\geq 1<\infty$ daje
$P(L)>0$.*
:::

::: remark
**Uwaga 6**. *Dodatnia gęstość $\pi_\beta$ (Warunek 1) jest warunkiem
**koniecznym**, ale **niewystarczającym** dla nieprzywiedlności.
Nieprzywiedlność zależy wyłącznie od kernela $q$ (Warunek 2), nie od
samej funkcji energii.*
:::

# Propozycje ruchów strukturalnych {#sec:proposals}

## Przesunięcie pojedynczej linii (Single Line Shift)

::: definition
**Definicja 16**. *Dla linii $L$ i skali $\sigma>0$:
$$q_{\mathrm{single}}(x'\mid x):\quad
  \theta'_L = (\theta_L+\epsilon)\bmod T,\quad\epsilon\sim\mathcal{N}(0,\sigma^2)$$
pozostałe współrzędne bez zmian. Kernel symetryczny.*
:::

## Przesunięcie komponentu (Component Move)

::: algorithm
$\mathrm{Queue}\leftarrow[L_{\mathrm{seed}}]$,
$\mathrm{Visited}\leftarrow\{L_{\mathrm{seed}}\}$ $\mathrm{Visited}$
:::

Propozycja: $\theta'_L=(\theta_L+\delta)\bmod T$ dla wszystkich
$L\in C$, gdzie $\delta\sim\mathcal{N}(0,\sigma^2)$.

Dobór parametru zaniku:
$$p_{\mathrm{decay}} = \min\!\left(0.5,\;\frac{0.8}{d_{\mathrm{avg}}}\right)$$
gdzie $d_{\mathrm{avg}}$ to średni stopień w $G_{\mathrm{dep}}$. Dla
$d\cdot p_{\mathrm{decay}}<1$ oczekiwany rozmiar komponentu jest
skończony.

## Ruch naprawczy (Smart Repair Move)

::: algorithm
Wybierz przesiadkę $\gamma\sim\mathrm{Uniform}(\Gamma)$ Niech $L_1,L_2$
= linie zaangażowane w $\gamma$ Wybierz linię
$L\sim\mathrm{Uniform}(\{L_1,L_2\})$
$\theta'_L=(\theta_L+\delta)\bmod T$
$H = \phi(\delta-\delta^*_{\mathrm{forward}};\;0,\sigma_{\mathrm{small}}^2)
  \;/\;\phi(\delta'-\delta^*_{\mathrm{backward}};\;0,\sigma_{\mathrm{small}}^2)$
$(x',H)$
:::

::: remark
**Uwaga 7**. *Przy tie-breaku (dwa rozwiązania o tej samej odległości na
okręgu) --- losowy wybór z $p=0.5$. Symetryczny, nie wpływa na Hastings
ratio.*
:::

## Swap offsetów

::: definition
**Definicja 17**. *Dla dwóch linii $L_1,L_2$ tej samej modalności:
$$q_{\mathrm{swap}}(x'\mid x):\quad \theta'_{L_1}=\theta_{L_2},\;\theta'_{L_2}=\theta_{L_1}$$
Kernel symetryczny (inwolucja).*
:::

## Mieszanka propozycji

$$q(x'\mid x) = \sum_{j=1}^{J}p_j\cdot q_j(x'\mid x),\qquad\sum_{j=1}^{J}p_j=1$$

  Propozycja           Waga $p_j$
  ------------------- ------------
  Single Line Shift       0.4
  Component Move          0.3
  Smart Repair            0.2
  Swap                    0.1

  : Sugerowane wagi początkowe

## Adaptacja wag propozycji {#sec:adaptation}

::: {#def:adaptation_sampling .definition}
**Definicja 18** (Adaptacja --- Tryb A (Sampling)). *Aktualizacja wag:
$$p_j^{(n+1)} = \Pi_\Delta\!\left(p_j^{(n)} + \gamma_n\cdot g(\hat{r}_j(n))\right)$$
gdzie $\Pi_\Delta$ to projekcja na sympleks,
$g(\hat{r}_j)=\hat{r}_j(1-\hat{r}_j)-\bar{g}$ preferuje umiarkowany
acceptance rate, $\gamma_n = n^{-a}$ z $a\in(1/2,1]$ (domyślnie
$a=0.7$).*

*Warunki gwarantujące poprawną stacjonarność:*

1.  ***Diminishing adaptation**: $\gamma_n\to 0$,
    $\sum_n\gamma_n=\infty$ $\Rightarrow$
    $\|p^{(n+1)}-p^{(n)}\|\leq C\gamma_n\to 0$.*

2.  ***Containment**: $p_j(n)\geq\varepsilon>0$ dla każdego $j$,
    $\sigma\in[\sigma_{\min},\sigma_{\max}]$.*

*Opcjonalnie: zamrożenie adaptacji po $N_0$ krokach.*
:::

::: definition
**Definicja 19** (Adaptacja --- Tryb B (Optimization)). *Ciągła
adaptacja bez wymogu *diminishing*. Algorytm nie jest formalnie
poprawnym MCMC. Output: $\arg\min$ po trajektorii (best-so-far).*
:::

# Inercyjny Parallel Tempering {#sec:pt}

## Parallel Tempering --- podstawy

Rozważamy $M$ replik łańcucha z temperaturami
$\beta_1<\beta_2<\cdots<\beta_M$.

Stan rozszerzony: $\mathbf{x}=(x^{(1)},\ldots,x^{(M)})\in\mathcal{X}^M$.

Rozkład docelowy:
$\Pi(\mathbf{x})=\prod_{i=1}^{M}\pi_{\beta_i}(x^{(i)})$.

Schemat temperatur (geometryczny):
$$\beta_i = \beta_1\cdot r^{i-1},\qquad
  r = \left(\frac{\beta_M}{\beta_1}\right)^{1/(M-1)}$$

## Mechanizm inercji {#sec:inertia}

::: definition
**Definicja 20** (Inercja). *Każda replika $i$ posiada wektor inercji
$I^{(i)}\in\mathbb{R}^{|\mathcal{L}|}_{\geq 1}$. Prawdopodobieństwo
wyboru linii $L$ do modyfikacji w replice $i$:
$$P^{(i)}(L) \propto \frac{1}{I^{(i)}_L}$$*
:::

  Poziom modalności    $I_L$
  ------------------- -------
  IC (dalekobieżny)     20
  Regionalny             5
  Lokalny                1

  : Parametry inercji

Konfiguracja replik: repliki "gorące" ($i\approx M$): $I^{(i)}_L=1$ dla
wszystkich $L$; repliki "zimne" ($i\approx 1$): $I^{(i)}_L$ według
tabeli.

## Algorytm

::: algorithm
Inicjalizuj $x^{(i)}$ losowo, $\beta_i$ geometrycznie, kalibruj
$M_{\mathrm{barrier}}$ $x^*$
:::

## Poprawność

::: theorem
**Twierdzenie 2**. *Inercyjny PT zachowuje poprawny rozkład
stacjonarny.*
:::

::: proof
*Proof.* $I_L$ jest stałe (nie zależy od $x$), więc prawdopodobieństwo
wyboru linii $L$ jest takie samo dla przejścia $x\to x'$ jak dla
$x'\to x$. Kernel pozostaje symetryczny (lub z obliczalnym Hastings
ratio). Każda replika $i$ próbkuje z $\pi_{\beta_i}$. Wymiana replik
spełnia szczegółową równowagę dla rozkładu produktowego $\Pi$. ◻
:::

# Analiza zbieżności i diagnostyka {#sec:convergence}

## Statystyka Gelman-Rubin

Dla $M$ niezależnych łańcuchów o długości $N$:
$$\hat{R}= \sqrt{\frac{\hat{V}}{W}},\qquad
  W=\frac{1}{M}\sum_{m=1}^{M}s_m^2,\quad
  B=\frac{N}{M-1}\sum_{m=1}^{M}(\bar{x}_m-\bar{x})^2,\quad
  \hat{V}=\frac{N-1}{N}W+\frac{1}{N}B$$ Kryterium zbieżności:
$\hat{R}<1.1$.

## Efektywna wielkość próby

$$\mathrm{ESS}= \frac{N}{1+2\sum_{k=1}^{\infty}\rho_k}$$ Kryterium:
$\mathrm{ESS}>100$.

## Tempo zbieżności

::: theorem
**Twierdzenie 3** (Tempo geometryczne). *Jeśli łańcuch jest
geometrycznie ergodyczny:
$\|P^n(x,\cdot)-\pi_\beta\|_{TV}\leq C(x)\cdot\rho^n$ dla $C(x)<\infty$,
$\rho<1$.*
:::

# Metryki jakości rozwiązania {#sec:metrics}

## Metryki operacyjne

  Metryka                  Definicja                        Cel
  ------------------------ ------------------------------- ------
  $\bar{w}(x)$             Średni ważony czas przesiadki    Min
  $\mathrm{MissRate}(x)$   \% przesiadek z $w<w_{\min}$     $=0$
  $\mathrm{LongRate}(x)$   \% przesiadek z $w>w_{\max}$     Min
  $\sigma_w(x)$            Odchylenie standardowe czasów    Min
  $w_{\max}(x)$            Najdłuższa przesiadka            Min

## Metryki robustności

-   $\mathrm{BufferAdequacy}(x)$ --- % przesiadek z buforem
    $\geq q_{0.9}$ opóźnienia

-   $\mathrm{CascadeRisk}(x)$ --- max zasięg propagacji opóźnienia
    w grafie

## Metryki diagnostyczne

::: {#def:peak_conc .definition}
**Definicja 21** (Peak concurrency).
*$$N^{\mathrm{peak}}_{\mathrm{conc}}(x) = \max_{t\in[0,T)}
    \sum_{\kappa\in\mathcal{K}}\mathbf{1}\{t\in[t^{\mathrm{dep}}_\kappa(x),\;
    t^{\mathrm{arr}}_\kappa(x))\}$$ Obliczalna
w $O(|\mathcal{K}|\log|\mathcal{K}|)$ algorytmem sweep-line. Raportowana
jako diagnostyka (dolne ograniczenie na flotę), nie wchodzi do
$\mathrm{Cost}$.*
:::

## Metryki pasażerskie

-   $\mathrm{WeightedJourneyTime}(x)$ --- średni ważony czas podróży
    end-to-end

-   $\mathrm{Accessibility}_T(x)$ --- % par OD osiągalnych w czasie
    $\leq T$

# Złożoność obliczeniowa {#sec:complexity}

## Koszt jednego kroku

  Operacja                                   Złożoność naiwna            Złożoność inkrementalna
  --------------------------------------- ----------------------- -------------------------------------
  Obliczenie $\mathrm{Cost}(x)$                $O(|\Gamma|)$                       ---
  $\Delta\mathrm{Cost}$ (single line)          $O(|\Gamma|)$                 $O(|\Gamma_L|)$
  $\Delta\mathrm{Cost}$ (component $C$)    $O(|C|\cdot|\Gamma|)$             $O(|\Gamma_C|)$
  $N^{\mathrm{peak}}_{\mathrm{conc}}$               ---            $O(|\mathcal{K}|\log|\mathcal{K}|)$

## Graf zależności

$G_{\mathrm{dep}}=(\mathcal{L},E_{\mathrm{dep}})$ gdzie
$(L_1,L_2)\in E_{\mathrm{dep}}$ jeśli istnieje przesiadka między $L_1$
i $L_2$. Dla propozycji zmieniającej linie w zbiorze $S$:
$$\Gamma_{\mathrm{affected}} = \{\gamma\in\Gamma:L_1(\gamma)\in S\;\text{lub}\;L_2(\gamma)\in S\}$$

## Równoległość

Dla PT z $M$ replikami na $M$ procesorach: lokalne kroki MH w pełni
równoległe, wymiana $O(M)$ komunikacji. Speedup bliski $M$ dla dużych
sieci.

# Kwestie do zbadania empirycznie {#sec:empirical}

Poniższe zagadnienia nie mogą być rozstrzygnięte teoretycznie i wymagają
eksperymentów.

## Mixing time i zbieżność

-   Ile kroków do zbieżności? ($\hat{R}<1.1$ na 4+ niezależnych
    łańcuchach)

-   Jaki $\mathrm{ESS}$/krok? (cel: $>0.01$)

-   Jak mixing skaluje się z rozmiarem sieci?

**Minimalny eksperyment**: 10 linii, 20 przesiadek, 4 łańcuchy, 100k
kroków.

## Porównanie propozycji ruchów

Która propozycja daje najlepszy $\mathrm{ESS}$/czas CPU? Optymalny mix
wag $p_j$? Czy Smart Repair przyspiesza zbieżność?

## Efektywność Parallel Tempering

Ile replik optymalnie ($M\in\{4,8,16,32\}$)? Jaki rozstaw temperatur
(cel: swap acceptance 20--30%)? Czy inercja pomaga?

## Kalibracja funkcji kosztu

Wrażliwość na $M$ (bariera), $\lambda_{\mathrm{fleet}}$,
$\lambda_{\mathrm{rob}}$. Analiza Pareto trade-offów.

## Benchmark vs alternatywy

  Metoda                 Cel porównania
  ---------------------- -----------------------------------
  Random search          Baseline minimum
  Greedy local search    Czy MCMC bije prostą heurystykę?
  Simulated annealing    Czy formalizm MCMC daje przewagę?
  MIP (małe instancje)   Jak daleko od optimum?
  Algorytm genetyczny    Porównanie z praktyką przemysłową

## Walidacja na rzeczywistych danych

Porównanie z GTFS, konsultacja z ekspertami, analiza wąskich gardeł.

## Skalowanie

Czas obliczeń vs $|\mathcal{L}|$ i $|\Gamma|$. Profiling. Speedup
inkrementalnego kosztu.

# Notacja {#app:notation}

                 Symbol                 Znaczenie
  ------------------------------------- ---------------------------------------------
              $\mathcal{M}$             Zbiór modalności
              $\mathcal{S}$             Zbiór stacji
              $\mathcal{L}$             Zbiór linii
              $\mathcal{K}$             Zbiór kursów
                $\Gamma$                Zbiór przesiadek
              $\mathcal{X}$             Przestrzeń stanów
            $x=(\theta_L)_L$            Stan (wektor offsetów)
                   $T$                  Wspólny okres ($=1440$ min)
                  $f_L$                 Częstotliwość linii $L$
               $N_L=T/f_L$              Liczba kursów linii $L$ na okres
              $w_\gamma(x)$             Czas oczekiwania na przesiadkę $\gamma$
           $\mathrm{Cost}(x)$           Całkowity koszt stanu $x$
               $\pi_\beta$              Rozkład Boltzmanna z temp. odwrotną $\beta$
              $q(x'\mid x)$             Kernel propozycji
              $\alpha(x,y)$             Prawdopodobieństwo akceptacji
              $\Psi(\cdot)$             Funkcja bariery (kary)
           $G_{\mathrm{dep}}$           Graf zależności linii
                  $I_L$                 Inercja linii $L$
   $N^{\mathrm{peak}}_{\mathrm{conc}}$  Peak concurrency (diagnostyka)

# Rejestr zmian względem wersji 1.0 {#app:changelog}

1.  **Jednolity okres**: $T_L=T=1440$ dla wszystkich linii
    (Założenie [1](#asm:uniform_period){reference-type="ref"
    reference="asm:uniform_period"}). Linie o różnych częstotliwościach
    mają $N_L=T/f_L$ kursów.

2.  **Przesiadki między liniami**: Nowa
    definicja [8](#def:transfer){reference-type="ref"
    reference="def:transfer"} --- przesiadka jako relacja
    $(L_1,s_1,L_2,s_2,\rho)$ między liniami, nie kursami. Czas
    oczekiwania (Def. [9](#def:waiting_time){reference-type="ref"
    reference="def:waiting_time"}) zawsze $\geq 0$ z definicji.

3.  **Ergodyczność --- poprawiony dowód**:
    Twierdzenie [1](#thm:ergodicity){reference-type="ref"
    reference="thm:ergodicity"} explicite rozdziela warunki na target
    ($\mathrm{Cost}<\infty$) i kernel ($P(L)>0$, wsparcie $\delta$ na
    całym $\mathbb{Z}$). Usunięta błędna implikacja "dodatnia gęstość
    $\Rightarrow$ nieprzywiedlność".

4.  **Robustność --- uczciwe proxy**:
    Definicja [13](#def:robustness){reference-type="ref"
    reference="def:robustness"} jawnie opisana jako heurystyczne proxy,
    nie formalna minimalizacja $\mathrm{CVaR}_\alpha$ straty.

5.  **Smart Repair --- reguła minimalnej normy**:
    Algorytm [\[alg:smart_repair\]](#alg:smart_repair){reference-type="ref"
    reference="alg:smart_repair"} definiuje $\delta^*$ jako przesunięcie
    o minimalnej wartości bezwzględnej. Tie-break losowy.

6.  **Adaptacja --- dwa tryby**: Tryb A
    (Definicja [18](#def:adaptation_sampling){reference-type="ref"
    reference="def:adaptation_sampling"}) z diminishing adaptation,
    containment ($p_j\geq\varepsilon$,
    $\sigma\in[\sigma_{\min},\sigma_{\max}]$). Tryb B --- ciągła
    adaptacja, jawnie nazwana metaheurystyką.

7.  **Proxy floty --- uczciwy opis**: $N_{\mathrm{fleet}}$ jako średnia
    intensywność (Little's Law). $N^{\mathrm{peak}}_{\mathrm{conc}}$
    (Definicja [21](#def:peak_conc){reference-type="ref"
    reference="def:peak_conc"}) jako diagnostyka, nie składnik kosztu.

8.  **Status metodologiczny**: Jawny podział na Tryb A (sampling)
    i Tryb B (optimization) ---
    Sekcja [1.3](#sec:status){reference-type="ref"
    reference="sec:status"}.
