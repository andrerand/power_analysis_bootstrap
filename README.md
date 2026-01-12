# A/B Test Power Analysis: Analytical vs Bootstrap Simulation

A comprehensive Jupyter notebook demonstrating two approaches to statistical power analysis for two-proportion A/B tests: analytical (closed-form) calculations and bootstrap/Monte Carlo simulation.

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Notebook Structure](#notebook-structure)
- [Use Cases](#use-cases)
- [Example Results](#example-results)
- [Key Concepts](#key-concepts)
- [Visualizations](#visualizations)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Educational Value](#educational-value)
- [Technical Details](#technical-details)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project provides an educational resource for understanding **statistical power analysis** in A/B testing contexts. It demonstrates two complementary approaches:

1. **Analytical (Closed-Form)**: Fast, exact calculations using mathematical formulas
2. **Bootstrap Simulation**: Flexible, simulation-based estimation that works for any test procedure

The notebook includes three advanced visualizations that reveal fundamental concepts in statistical inference, helping you build intuition about power, p-values, and sampling variability.

## Features

- 📊 **Analytical Sample Size Calculation** - Closed-form formula for two-proportion z-test
- 🎲 **Bootstrap/Monte Carlo Simulation** - Flexible power estimation through simulation
- 🔍 **Sample Size Search Algorithm** - Iterative grid search to find minimum required n
- 📈 **Three Advanced Visualizations**:
  - **P-value Distribution** (null vs alternative hypothesis)
  - **Power Convergence Analysis** (precision vs simulation count)
  - **Effect Size Distribution** (sampling variability demonstration)
- ✅ **Comparison Framework** - Validate simulation results against analytical calculations

## Installation

```bash
# Clone repository
git clone https://github.com/andrerand/power_analysis_bootstrap.git
cd power_analysis_bootstrap

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter notebook
cd Scripts
jupyter notebook power_analysis.ipynb
```

## Quick Start

### Analytical Approach (Fast)

```python
import numpy as np
from scipy.stats import norm

def analytical_sample_size(p1, p2, alpha=0.05, power=0.80):
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    effect = abs(p2 - p1)
    variance_sum = p1*(1-p1) + p2*(1-p2)
    n = ((z_alpha + z_beta)**2 * variance_sum) / (effect**2)
    return int(np.ceil(n))

# Calculate required sample size
n = analytical_sample_size(
    p1=0.0129,      # Baseline: 1.29%
    p2=0.013545,    # Treatment: 1.3545% (5% relative lift)
    alpha=0.05,     # 95% confidence
    power=0.80      # 80% power
)
print(f"Required sample size: {n:,} per group")
# Output: Required sample size: 492,321 per group
```

### Bootstrap Simulation Approach (Flexible)

```python
def simulate_power_proportions(baseline_rate, treatment_rate, n_per_group,
                               n_simulations=5000, alpha=0.05):
    significant_count = 0
    np.random.seed(42)

    for _ in range(n_simulations):
        control = np.random.binomial(1, baseline_rate, n_per_group)
        treatment = np.random.binomial(1, treatment_rate, n_per_group)
        # ... perform two-proportion z-test ...
        if p_value < alpha:
            significant_count += 1

    return significant_count / n_simulations

# Estimate power
power = simulate_power_proportions(0.0129, 0.013545, 492000)
print(f"Estimated power: {power:.1%}")
# Output: Estimated power: 80.2%
```

## Notebook Structure

The notebook contains **19 cells** organized into three main parts:

### Part 1: Analytical Approach (Cells 0-5)
- Introduction and mathematical formula
- Analytical sample size function
- Example calculation and interpretation

### Part 2: Bootstrap Simulation (Cells 6-11)
- Theory and intuition behind bootstrap methods
- Power simulation function
- Minimum sample size search algorithm
- Comparison with analytical results
- Power curve visualization

### Part 3: Advanced Visualizations (Cells 12-18)
- Helper function for detailed simulations
- **Visualization 1**: P-value distribution under null vs alternative
- **Visualization 2**: Power estimate convergence analysis
- **Visualization 3**: Effect size sampling distribution

## Use Cases

### When to Use Analytical Approach

✅ Standard statistical tests (two-proportion z-test, t-test, etc.)
✅ Speed is important
✅ Want exact results with no sampling variability
✅ Understand mathematical assumptions

### When to Use Bootstrap Simulation

✅ No closed-form power formula exists for your test
✅ Complex experimental designs (stratification, clustering)
✅ Want to validate analytical results
✅ Need to test non-standard assumptions or distributions

## Example Results

For an A/B test with:
- **Baseline conversion**: 1.29%
- **Treatment conversion**: 1.3545% (5% relative lift)
- **Significance level**: α = 0.05 (95% confidence)
- **Target power**: 80%

**Required sample size**: ~492,000 per group (984,000 total observations)

Both analytical and simulation approaches converge on this result, demonstrating their equivalence for standard tests while highlighting the flexibility of simulation for more complex scenarios.

## Key Concepts

### Statistical Power
The probability of correctly detecting a true effect. Typically set to 80%, meaning an 80% chance of finding a statistically significant result when the treatment truly works.

### Type I Error (α)
False positive rate—probability of declaring significance when no effect exists. Typically set to 5% (α = 0.05).

### Type II Error (β)
False negative rate—probability of missing a true effect. Related to power: β = 1 - power.

### Effect Size
The magnitude of the difference you want to detect. Smaller effects require larger samples to detect reliably.

### Bootstrap/Monte Carlo Simulation
A computational approach that estimates statistical properties by "replaying" the experiment thousands of times with known parameters, then observing how often the statistical test succeeds.

## Visualizations

### 1. P-value Distribution (Null vs Alternative)

Demonstrates the fundamental concept behind statistical power by showing p-value distributions:
- **Under null hypothesis**: Uniformly distributed (0 to 1), ~5% below α = 0.05
- **Under alternative hypothesis**: Concentrated near zero, ~80% below α = 0.05

**Key insight**: Power = probability that p < α when effect is real.

### 2. Power Convergence Curve

Shows how estimated power stabilizes as simulation count increases:
- Tests 100 to 10,000 simulations
- Displays 95% confidence bands
- Demonstrates precision vs. computation trade-off
- Standard error decreases as √(1/n_simulations)

**Key insight**: More simulations → more precision, but diminishing returns.

### 3. Effect Size Distribution

Reveals sampling variability in observed effect sizes:
- Histogram of 5,000 simulated effect size observations
- Overlays theoretical normal distribution
- Shows 95% confidence interval
- Compares observed mean to true effect

**Key insight**: Even with large samples, individual experiments vary due to randomness.

## Requirements

- **Python**: 3.12+
- **Packages**:
  - `numpy` - Numerical computations
  - `scipy` - Statistical functions
  - `matplotlib` - Plotting
  - `pandas` - Data manipulation
  - `seaborn` - Statistical visualizations
  - `jupyter` - Notebook environment
  - `ipykernel` - Jupyter kernel

**Computation time**: Running all visualizations takes approximately 5-10 minutes.

## Project Structure

```
power_analysis_bootstrap/
├── Scripts/
│   └── power_analysis.ipynb    # Main notebook (19 cells)
├── Images/
│   └── sample-size-formula.png # Formula visualization
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore patterns
└── README.md                  # This file
```

## Educational Value

This notebook helps you understand:

- 🎯 **Analytical vs Simulation Trade-offs** - When to use each approach and why
- 📏 **Sample Size Requirements** - Why detecting small effects requires large samples
- 📊 **P-value Behavior** - How p-values differ under null vs alternative hypotheses
- 🔄 **Simulation Precision** - The relationship between simulation count and estimate accuracy
- 📈 **Sampling Variability** - Why individual experiments give different results even with adequate power

The visualizations make abstract statistical concepts concrete by showing actual distributions and convergence patterns.

## Technical Details

### Statistical Test
Two-proportion z-test with pooled variance:

```
z = (p_treatment - p_control) / SE
SE = √[p_pooled × (1 - p_pooled) × (2/n)]
p_pooled = (n_control×p_control + n_treatment×p_treatment) / (2n)
```

### Reproducibility
All simulations use `np.random.seed(42)` for reproducibility.

### Sample Size Search
Grid search algorithm with configurable:
- Start point (400,000)
- Step size (20,000)
- Maximum (600,000)

### Confidence Bands
95% confidence intervals calculated as: estimate ± 1.96 × SE

Where SE = √[power × (1-power) / n_simulations]

## License

MIT License - Feel free to use this for educational purposes.

## Acknowledgments

This project demonstrates statistical power analysis concepts from classical hypothesis testing literature. The bootstrap/Monte Carlo approach follows simulation methodologies established in computational statistics.

**Co-authored by**: Claude Sonnet 4.5 <noreply@anthropic.com>

---

**Questions or suggestions?** Open an issue on [GitHub](https://github.com/andrerand/power_analysis_bootstrap/issues).
