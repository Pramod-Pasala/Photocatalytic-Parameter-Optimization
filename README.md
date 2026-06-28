# Photocatalytic Parameter Optimization

Bayesian optimization framework for finding optimal photocatalytic experiment parameters using Gaussian Process Regression, feature importance analysis, and multi-objective Pareto optimization.

## Overview

This project applies machine learning to photocatalysis research — a domain where experiments are expensive and time-consuming. By using Bayesian Optimization (BO) with Gaussian Process Regression (GPR), the system identifies optimal experimental parameters (e.g., catalyst dosage, pH, light intensity) while minimizing the number of physical experiments needed.

## Key Features

- **Bayesian Optimization**: Uses Gaussian Process Regression as a surrogate model with Expected Improvement (EI) acquisition function to guide experiment selection
- **Feature Importance Analysis**: Combines SHAP values and permutation importance to identify which parameters most influence photocatalytic efficiency
- **Multi-Objective Optimization**: Generates Pareto-optimal solution sets across multiple optimization strategies (4 options included as Excel outputs)
- **Visualization**: Built-in plotting for convergence curves, parameter importance, and Pareto fronts

## Tech Stack

- **Python 3** with `uv` for dependency management
- **scikit-learn** — Gaussian Process Regression, Random Forest, cross-validation
- **SciPy** — Optimization and statistical functions
- **SHAP** — Model interpretability via SHAP values
- **Matplotlib** — Visualization
- **Docker** — Reproducible environment via devcontainer

## Project Structure

```
Photocatalytic-Parameter-Optimization/
├── BOMo.py                              # Main Bayesian Optimization module
├── multi_objective_optimization.ipynb   # Jupyter notebook for multi-objective analysis
├── input.xlsx                           # Input experimental data
├── option_1_pareto_optimal_solutions.xlsx  # Pareto solutions - Strategy 1
├── option_2_pareto_optimal_solutions.xlsx  # Pareto solutions - Strategy 2
├── option_3_pareto_optimal_solutions.xlsx  # Pareto solutions - Strategy 3
├── option_4_pareto_optimal_solutions.xlsx  # Pareto solutions - Strategy 4
├── requirements.txt                     # Python dependencies
├── pyproject.toml                       # Project configuration
├── Dockerfile                           # Container definition
└── .devcontainer/                       # VS Code dev container setup
```

## Setup

### Using uv (recommended)

```bash
uv sync
uv run python BOMo.py
```

### Using pip

```bash
pip install -r requirements.txt
python BOMo.py
```

### Using Docker

```bash
docker build -t photocatalytic-opt .
docker run -it photocatalytic-opt
```

## Usage

1. Place your experimental data in `input.xlsx`
2. Run the optimization:
   ```bash
   python BOMo.py
   ```
3. For multi-objective analysis, open the notebook:
   ```bash
   jupyter notebook multi_objective_optimization.ipynb
   ```
4. Results are saved as Pareto-optimal solution sets in `option_*.xlsx` files

## Optimization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_INIT` | 7 | Initial random seed points |
| `N_ITER` | 20 | Number of BO iterations |
| `XI` | 0.1 | Exploration-exploitation trade-off (lower = exploit) |
| `N_RESTARTS` | 20 | GP optimization restarts |

## License

MIT License — see [LICENSE](LICENSE) for details.
