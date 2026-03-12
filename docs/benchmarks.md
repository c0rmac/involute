# SO(d) Solver Benchmarks

## Table of Contents
- [Ackley Function](#ackley-function)
    - [Dimension 3](#ackley---dimension-3)
    - [Dimension 5](#ackley---dimension-5)
    - [Dimension 10](#ackley---dimension-10)
    - [Dimension 20](#ackley---dimension-20)
    - [Dimension 50](#ackley---dimension-50)
- [Schwefel Function](#schwefel-function)
    - [Dimension 3](#schwefel---dimension-3)
    - [Dimension 5](#schwefel---dimension-5)

---

## Ackley Function


$$ f(X) = -a \exp\left(-b \sqrt{\frac{1}{n} \sum (X - I)^2}\right) - \exp\left(\frac{1}{n} \sum \cos(c (X - I))\right) + a + \exp(1) $$

Where $a=20$, $b=0.2$, $c=2\pi$, and $n=d^2$.

### Ackley - Dimension 3

| Type: Aggressive | Type: ExtraSafe |
| --- | --- |
| ![ackley d=3 type=Aggressive](./analysis/results/ackley_viz/ackley_d=3_type=Aggressive.jpg) | ![ackley d=3 type=ExtraSafe](./analysis/results/ackley_viz/ackley_d=3_type=ExtraSafe.jpg) |
| **Runs:** 100<br>**Success:** 100.0%<br>**Mean Steps:** 29.8<br>**Median Steps:** 30.0 | **Runs:** 100<br>**Success:** 100.0%<br>**Mean Steps:** 79.0<br>**Median Steps:** 80.0 |

---

### Ackley - Dimension 5

| Type: Aggressive | Type: ExtraSafe |
| --- | --- |
| ![ackley d=5 type=Aggressive](./analysis/results/ackley_viz/ackley_d=5_type=Aggressive.jpg) | ![ackley d=5 type=ExtraSafe](./analysis/results/ackley_viz/ackley_d=5_type=ExtraSafe.jpg) |
| **Runs:** 50<br>**Success:** 100.0%<br>**Mean Steps:** 78.5<br>**Median Steps:** 79.0 | **Runs:** 100<br>**Success:** 97.0%<br>**Mean Steps:** 259.2<br>**Median Steps:** 260.0 |

---

### Ackley - Dimension 10

| Type: Aggressive | Type: ExtraSafe |
| --- | --- |
| ![ackley d=10 type=Aggressive](./analysis/results/ackley_viz/ackley_d=10_type=Aggressive.jpg) | ![ackley d=10 type=ExtraSafe](./analysis/results/ackley_viz/ackley_d=10_type=ExtraSafe.jpg) |
| **Runs:** 50<br>**Success:** 98.0%<br>**Mean Steps:** 267.9<br>**Median Steps:** 268.0 | **Runs:** 100<br>**Success:** 95.0%<br>**Mean Steps:** 1286.2<br>**Median Steps:** 1421.0 |

---

### Ackley - Dimension 20

| Type: Aggressive | Type: ExtraSafe |
| --- | --- |
| ![ackley d=20 type=Aggressive](./analysis/results/ackley_viz/ackley_d=20_type=Aggressive.jpg) | ![ackley d=20 type=ExtraSafe](./analysis/results/ackley_viz/ackley_d=20_type=ExtraSafe.jpg) |
| **Runs:** 50<br>**Success:** 96.0%<br>**Mean Steps:** 458.8<br>**Median Steps:** 458.0 | **Runs:** 50<br>**Success:** 76.0%<br>**Mean Steps:** 3558.7<br>**Median Steps:** 3546.5 |

---

### Ackley - Dimension 50

| Type: ExtraSafe |
| --- |
| ![ackley d=50 type=ExtraSafe](./analysis/results/ackley_viz/ackley_d=50_type=ExtraSafe.jpg) |
| **Runs:** 50<br>**Success:** 96.0%<br>**Mean Steps:** 20380.8<br>**Median Steps:** 20384.5 |

---

## Schwefel Function


$$ f(X) = 418.9829n - \sum Z \sin(\sqrt{|Z|}) $$

Where $Z = 250(X - I) + 420.968746$, and $n=d^2$.

### Schwefel - Dimension 3

| Type: Custom |
| --- |
| ![schwefel d=3 type=Custom](./analysis/results/schwefel_viz/schwefel_d=3_type=Custom.jpg) |
| **Runs:** 50<br>**Success:** 92.0%<br>**Mean Steps:** 50.0<br>**Median Steps:** 52.0 |

---

### Schwefel - Dimension 5

| Type: Custom |
| --- |
| ![schwefel d=5 type=Custom](./analysis/results/schwefel_viz/schwefel_d=5_type=Custom.jpg) |
| **Runs:** 50<br>**Success:** 90.0%<br>**Mean Steps:** 776.8<br>**Median Steps:** 778.0 |

---

