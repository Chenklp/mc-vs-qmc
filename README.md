# Monte Carlo-vs-Quasi-Monte Carlo
Uncertainty Quantification assignment with Monte Carlo and Quasi-Monte Carlo
# mc-vs-qmc

This project implements Uncertainty Quantification (UQ) for a linear damped oscillator using two sampling methods:

- Standard Monte Carlo (MC)
- Quasi-Monte Carlo (QMC) based on Halton sequences

## Problem Description

We solve the second-order differential equation of a damped oscillator:

$$
\frac{d^2y}{dt^2} + c \frac{dy}{dt} + k y = f \cos(\omega t)
$$

The uncertainty lies in the frequency parameter $\omega$, assumed to follow a uniform distribution $\mathcal{U}(0.95, 1.05)$.  
The goal is to estimate the mean and variance of the displacement $y(t)$ at time $t=10$ using both MC and QMC methods, and compare their convergence behavior.

The true reference values are:
- Mean: -0.43893703  
- Variance: 0.00019678  

These were computed using 1,000,000 samples and are provided in `oscillator_ref.txt`.

excacly
We consider the linear damped oscillator model defined as:

$$
\begin{cases}
\frac{d^2y}{dt^2}(t) + c \frac{dy}{dt}(t) + ky(t) = f \cos(\omega t) \\
y(0) = y_0 \\
\frac{dy}{dt}(0) = y_1
\end{cases}
$$

where:
- $c$ is the damping coefficient  
- $k$ is the spring constant  
- $f$ is the forcing amplitude  
- $\omega$ is the frequency  
- $y_0$ is the initial position  
- $y_1$ is the initial velocity  

We solve this system for $t \in [0, 10]$ using a time step of $\Delta t = 0.01$.

---

### Deterministic Case

First, consider the deterministic setting (from *Tutorial 1*) where all parameters are fixed:

$$
c = 0.5, \quad k = 2.0, \quad f = 0.5, \quad \omega = 1.0, \quad y_0 = 0.5, \quad y_1 = 0.0
$$

---

### Uncertain Frequency

Now consider a more realistic case, where the value of $\omega$ is uncertain.  
We assume:

$$
\omega \sim \mathcal{U}(0.95,\ 1.05)
$$

The goal is to propagate this uncertainty through the system using two methods:

- Standard Monte Carlo sampling  
- Quasi-Monte Carlo sampling using Halton sequences

---

### Reference Solution

Since the true statistics are not analytically known, we compare to a reference solution provided in `oscillator_ref.txt`:

$$
\mathbb{E}_{\mathrm{ref}}[y(10)] = -0.43893703, \quad \mathbb{V}_{\mathrm{ref}}[y(10)] = 0.00019678
$$

This reference was computed using $N_{\mathrm{ref}} = 1000000$ samples and does not need to be recomputed.




## Files

- `assign4.py`: main simulation and plotting script  
- `oscillator.py`: defines the oscillator model  
- `oscillator_ref.txt`: contains the reference statistics  
- `convergence.png`: plot of convergence results  
- `trajectory.png`: 10 sample trajectories using MC


