# KQuant

A modular Python library for quantitative finance, focused on derivative pricing, stochastic modelling, risk analysis and calibration methods.

KQuant provides an extensible framework where financial products, stochastic models and numerical pricing methods are separated through an object-oriented architecture.

---

# Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Financial Products](#financial-products)
- [Stochastic Models](#stochastic-models)
- [Pricing Engines](#pricing-engines)
- [Greeks](#greeks)
- [Calibration](#calibration)
- [Examples](#examples)
- [Tests](#tests)
- [Future Improvements](#future-improvements)

---

# Overview

KQuant is a Python quantitative finance library designed for derivative pricing, risk analysis and stochastic modelling.

The objective is to provide a modular framework implementing classical and advanced quantitative finance methods:

- closed-form option pricing
- Monte Carlo simulation
- stochastic volatility modelling
- Heston Fourier pricing
- Greeks computation
- implied volatility calculation
- volatility surface construction
- stochastic model calibration

The architecture separates financial products, models and pricing engines, allowing components to be extended independently.

---

# Features

## Derivative Products

Implemented products:

- European options
- Asian options
- Barrier options

## Pricing Methods

Implemented pricing engines:

- Black-Scholes analytical pricing
- Monte Carlo simulation
- Heston Fourier semi-analytical pricing

## Risk Management

Implemented Greeks:

- Delta
- Gamma
- Vega
- Theta
- Rho

Available approaches:

- analytical Greeks
- finite difference Greeks

## Calibration

Implemented tools:

- implied volatility calculation
- price surface construction
- volatility surface construction
- Heston calibration

---

# Installation

## Clone the repository

```bash
git clone https://github.com/AlexyKLEIN/KQuant.git
cd KQuant
```

## Install the package

```bash
pip install -e .[test]
```

## Requirements

- Python >= 3.10
- NumPy
- SciPy

---

# Project Structure

```text
KQuant/
│
├── README.md
├── pyproject.toml
├── .gitignore
│
├── examples/
│   ├── 01_black_scholes_pricing.py
│   ├── 02_european_greeks.py
│   ├── 03_monte_carlo_pricing.py
│   ├── 04_asian_option_pricing.py
│   ├── 05_barrier_option_pricing.py
│   ├── 06_asian_option_greeks.py
│   ├── 07_barrier_option_greeks.py
│   ├── 08_heston_fourier_pricing.py
│   ├── 09_heston_monte_carlo_pricing.py
│   ├── 10_implied_volatility.py
│   ├── 11_price_surface.py
│   ├── 12_volatility_surface.py
│   ├── 13_heston_calibration_price.py
│   └── 14_heston_calibration_iv.py
│
├── KQuant/
│   │
│   ├── calibration/
│   │   ├── implied_volatility.py
│   │   ├── price_surface.py
│   │   ├── volatility_surface.py
│   │   ├── heston_calibrator.py
│   │   └── __init__.py
│   │
│   ├── greek/
│   │   ├── analytic.py
│   │   ├── finite_difference.py
│   │   └── __init__.py
│   │
│   ├── market/
│   │   ├── market_data.py
│   │   └── __init__.py
│   │
│   ├── models/
│   │   ├── model.py
│   │   ├── black_and_scholes.py
│   │   ├── heston.py
│   │   └── __init__.py
│   │
│   ├── pricing/
│   │   ├── pricing_engine.py
│   │   ├── analytic.py
│   │   ├── monte_carlo.py
│   │   ├── heston_fourier.py
│   │   └── __init__.py
│   │
│   ├── products/
│   │   ├── product.py
│   │   ├── european_option.py
│   │   ├── asian_option.py
│   │   ├── barrier_option.py
│   │   └── __init__.py
│   │
│   ├── tools/
│   │   ├── normal.py
│   │   └── __init__.py
│   │
│   └── __init__.py
│
└── tests/
    ├── conftest.py
    ├── test_engines.py
    ├── test_greeks.py
    ├── test_calibration.py
    └── __init__.py
```

---

# Quick Start

Run an example:

```bash
python examples/01_black_scholes_pricing.py
```

Available examples demonstrate:

- European option pricing
- Monte Carlo pricing
- Asian options
- Barrier options
- Greeks computation
- Heston pricing
- Implied volatility
- Surface construction
- Model calibration

---

# Architecture

KQuant follows a modular object-oriented architecture.

The main components are:

- Market data
- Financial products
- Stochastic models
- Pricing engines
- Greeks calculators
- Calibration tools

The same product can be priced using different models and numerical methods.

Example:

```
European Option

        |
        +---- Black-Scholes + Analytical Engine
        |
        +---- Black-Scholes + Monte Carlo Engine
        |
        +---- Heston + Monte Carlo Engine
        |
        +---- Heston + Fourier Engine
```

---

# Financial Products

The `products` module defines derivative contracts independently from pricing methods.

Each product contains:

- contract parameters
- payoff definition
- product-specific behaviour

Implemented products:

| Product | Description | Pricing |
|---|---|---|
| European Option | Vanilla call and put | Analytical, Monte Carlo, Fourier |
| Asian Option | Arithmetic average option | Monte Carlo |
| Barrier Option | Path-dependent option | Monte Carlo |

---

# Stochastic Models

The `models` module contains stochastic processes describing the underlying asset.

Implemented models:

- Black-Scholes
- Heston

---

## Black-Scholes Model

The Black-Scholes model assumes constant volatility.

$$
dS_t=rS_tdt+\sigma S_tdW_t
$$

Parameters:

| Parameter | Description |
|---|---|
| r | Risk-free rate |
| sigma | Volatility |

Used for:

- analytical pricing
- Monte Carlo simulation
- implied volatility
- analytical Greeks

---

## Heston Model

The Heston model introduces stochastic volatility.

$$
dS_t=rS_tdt+\sqrt{v_t}S_tdW_t
$$

$$
dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t
$$

Parameters:

| Parameter | Description |
|---|---|
| v0 | Initial variance |
| kappa | Mean reversion speed |
| theta | Long-term variance |
| xi | Volatility of volatility |
| rho | Correlation |

Used for:

- Monte Carlo simulation
- Fourier pricing
- calibration

---

# Pricing Engines

The `pricing` module contains numerical methods used to compute derivative prices.

Implemented engines:

| Engine | Method |
|---|---|
| AnalyticEngine | Closed-form pricing |
| MonteCarloEngine | Path simulation |
| HestonFourierEngine | Fourier inversion |

---

## Analytic Engine

Provides closed-form formulas.

Currently supports:

- Black-Scholes model
- European options

Example:

```python
from KQuant.pricing import AnalyticEngine

engine = AnalyticEngine()

price = engine.price(
    market,
    model,
    option
)
```

---

## Monte Carlo Engine

The Monte Carlo engine estimates prices by simulation.

$$
Price=e^{-rT}E[Payoff]
$$

Parameters:

- `n_paths`: number of simulated paths
- `n_steps`: number of time steps

Supported products:

- European options
- Asian options
- Barrier options

---

## Heston Fourier Engine

The Heston Fourier engine uses numerical inversion of the characteristic function.

Advantages:

- fast pricing
- no simulation noise
- suitable for calibration

Supported:

- European options

---

# Greeks

KQuant provides option sensitivities.

Implemented Greeks:

- Delta
- Gamma
- Vega
- Theta
- Rho

---

## Analytical Greeks

Closed-form Greeks are available for:

- Black-Scholes
- European options

---

## Finite Difference Greeks

Numerical Greeks are computed by perturbing:

- spot price
- volatility
- interest rate
- maturity

Used for:

- Asian options
- Barrier options
- products without closed-form solutions

---

# Calibration

The calibration module provides tools to extract market information and estimate model parameters.

---

## Implied Volatility

Computes the volatility consistent with an observed option price.

Supported:

- European options
- Black-Scholes model

---

## Price Surface

Stores option prices across:

- strikes
- maturities

Used as calibration input.

---

## Volatility Surface

Stores implied volatilities across:

- strikes
- maturities

---

## Heston Calibration

The Heston calibrator estimates the model parameters:

- initial variance \(v_0\)
- mean reversion speed \(\kappa\)
- long-term variance \(\theta\)
- volatility of volatility \(\xi\)
- correlation between asset and volatility processes \(\rho\)

The calibration can be performed using different market targets:

- option market prices
- implied volatilities

The calibration can be performed using either a custom optimization approach or SciPy optimization algorithms for faster convergence.

The objective is to minimize the difference between model outputs and market observations.

---

# Examples

The `examples` directory contains executable demonstrations.

| Example | Description |
|---|---|
| 01 | Black-Scholes pricing |
| 02 | European Greeks |
| 03 | Monte Carlo pricing |
| 04 | Asian option pricing |
| 05 | Barrier option pricing |
| 06 | Asian option Greeks |
| 07 | Barrier option Greeks |
| 08 | Heston Fourier pricing |
| 09 | Heston Monte Carlo pricing |
| 10 | Implied volatility |
| 11 | Price surface |
| 12 | Volatility surface |
| 13 | Heston calibration using prices |
| 14 | Heston calibration using implied volatility |

---

# Tests

KQuant uses `pytest` for automated validation.

Tests cover:

- pricing engines
- Greeks computation
- calibration methods

Run tests:

```bash
pytest
```

*Note for Windows users: If the `pytest` command is not recognized, use:*
```bash
python -m pytest
```

Validation includes comparisons between:

- analytical solutions
- Monte Carlo estimations
- QuantLib benchmarks

---

# Future Improvements

Possible extensions:

- American options
- local volatility models
- SABR model
- automatic differentiation Greeks
- variance reduction techniques
- improved calibration algorithms
- GPU acceleration for Monte Carlo