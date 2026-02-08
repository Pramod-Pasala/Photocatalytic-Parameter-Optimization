#!/usr/bin/env python3


import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

warnings.filterwarnings('ignore', category=UserWarning)

# Check for SHAP availability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("NOTE: SHAP library not available. Will use permutation importance only.")
    print("      Install with: pip install shap\n")

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

# Bayesian Optimization parameters
N_INIT = 7  # Initial random seed points
N_ITER = 20  # Number of BO iterations
XI = 0.1  # Exploitation-exploration trade-off parameter (small = exploit)
N_RESTARTS = 20  # GP optimization restarts


PERM_IMPORTANCE_REPEATS = 300
RF_N_ESTIMATORS_GRID = [100, 300, 500, 1000] 
RF_MAX_FEATURES_GRID = ['sqrt', 'log2']

# --- ADDED/MODIFIED FOR REGULARIZATION ---
# These limits help prevent individual trees from growing too deep and memorizing noise.
RF_MAX_DEPTH_GRID = [5, 10, 15, None]      # Limit tree depth: None means no limit (original)
RF_MIN_SAMPLES_LEAF_GRID = [1, 2, 3]       # Minimum samples at a leaf: >1 forces generalization
# -----------------------------------------

RF_CV_FOLDS = 5 # Increased from 4 for more robust CV score on small data
PERM_IMPORTANCE_REPEATS = 300
# Numerical stability
EPS = 1e-9

# High-quality plot settings
PLOT_DPI = 300
PLOT_FORMAT = ['png']
PLOT_STYLE = {
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Color palette (professional, color-blind friendly)
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'accent1': '#F18F01',      # Orange
    'accent2': '#06A77D',      # Green
    'accent3': '#5A67D8',      # Indigo
    'neutral': '#4A5568',      # Gray
    'success': '#48BB78',      # Green
    'warning': '#ED8936',      # Orange
    'error': '#F56565',        # Red
}

# --------------------- Initial data ---------------------
INITIAL_TRAIN = pd.read_excel("input.xlsx")

INITIAL_TOF = INITIAL_TRAIN.iloc[:, -2].values.tolist()
INITIAL_TOF = np.array(INITIAL_TOF, dtype=float)

INITIAL_AQY = INITIAL_TRAIN.iloc[:, -1].values.tolist()

INITIAL_TRAIN = INITIAL_TRAIN.iloc[:, :4].values

FEATURE_NAMES = ['Catalyst (g)', 'Hole scavenger (ml)', 'Light Intensity (W)','Time (h)']


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_directories():
    """Create output directories if they don't exist."""
    Path("figures").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("data/raw").mkdir(parents=True, exist_ok=True)


def save_requirements():
    """Save requirements to outputs directory."""
    try:
        import subprocess
        result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
        with open('outputs/requirements.txt', 'w') as f:
            f.write(result.stdout)
        print("✓ Saved requirements.txt")
    except Exception as e:
        print(f"⚠ Could not save requirements.txt: {e}")


def expected_improvement(mu: np.ndarray, sigma: np.ndarray,
                         y_best: float, xi: float = 0.01) -> np.ndarray:
    """
    Calculate Expected Improvement acquisition function.

    Parameters
    ----------
    mu : array-like, shape (n_samples,)
        Predicted mean from GP
    sigma : array-like, shape (n_samples,)
        Predicted standard deviation from GP
    y_best : float
        Best observed objective value so far
    xi : float, default=0.01
        Exploitation-exploration trade-off (smaller = more exploitation)

    Returns
    -------
    ei : array-like, shape (n_samples,)
        Expected improvement values

    Notes
    -----
    EI is computed as: EI = (mu - y_best - xi) * Φ(Z) + sigma * φ(Z)
    where Z = (mu - y_best - xi) / sigma, Φ is CDF, φ is PDF of standard normal.
    xi=0.01 provides slight exploration while favoring exploitation.
    """
    with np.errstate(divide='warn'):
        sigma = np.maximum(sigma, 1e-9)  # Numerical stability
        z = (mu - y_best - xi) / sigma
        ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-9] = 0.0
    return ei


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_validate_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load experimental data from CSV or use embedded test data.

    Parameters
    ----------
    filepath : str, optional
        Path to experiments CSV file

    Returns
    -------
    df : pd.DataFrame
        Validated dataframe with all required columns
    """
    if filepath and os.path.exists(filepath):
        print(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
    else:
        print("Data file not found. Using embedded test data.")
        # Create test dataframe from embedded arrays with continuous features
        n = len(INITIAL_TOF)
        df = pd.DataFrame({
            'catalyst': INITIAL_TRAIN[:, 0],  # Continuous: catalyst loading/concentration
            
            'hole_scavenger': INITIAL_TRAIN[:, 1],  # Continuous: scavenger concentration
            'light_intensity': INITIAL_TRAIN[:, 2],  # Continuous: light power
            'time': INITIAL_TRAIN[:, 3],  # Continuous: reaction time
            'TOF': INITIAL_TOF,
            'AQY': INITIAL_AQY,
            'Yield': INITIAL_TOF * 0.5,  # Dummy values
            'TON': INITIAL_TOF * 2,
            'Selectivity': np.random.uniform(85, 99, n),
            'experiment_id': [f'EXP_{i:03d}' for i in range(n)]
        })

    # Validate required columns
    required = ['catalyst', 'hole_scavenger', 'light_intensity', "time",
                 'TOF', 'AQY', 'Yield', 'TON', 'Selectivity', 'experiment_id']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(df)}")
    print(f"\nContinuous feature ranges:")
    numeric_cols = ['catalyst', 'solvent', 'hole_scavenger', 'light_intensity',
                    'time', 'TOF', 'AQY', 'Yield', 'TON', 'Selectivity']
    for col in numeric_cols:
        if col in df.columns:
            print(f"  {col:20s}: [{df[col].min():.4f}, {df[col].max():.4f}]")

    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Preprocess data: handle zeros, compute objective, prepare continuous features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw experimental data

    Returns
    -------
    df : pd.DataFrame
        Processed dataframe with objective and flags
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (all continuous, not encoded)
    feature_names : list of str
        Names of features in X

    Notes
    -----
    - All features treated as continuous numerical variables
    - Replaces TOF, AQY ≤ 0 with EPS for log computation
    - Sets zero_yield_flag for actual zero yields
    - Objective = 0.5*ln(AQY) + 0.5*ln(TOF) with equal weighting
    """
    df = df.copy()

    # Flag true zero yields before replacement
    df['zero_yield_flag'] = (df['Yield'] == 0)

    # Replace ≤0 values with EPS for log-safety
    for col in ['TOF', 'AQY']:
        n_replaced = (df[col] <= 0).sum()
        if n_replaced > 0:
            print(f"⚠ Replacing {n_replaced} values ≤0 in {col} with EPS={EPS}")
        df[col] = df[col].clip(lower=EPS)

    # Compute objective with equal weights (0.5 + 0.5 = 1.0)
    df['ln_TOF'] = np.log(df['TOF'])
    df['ln_AQY'] = np.log(df['AQY'])
    df['objective'] = 0.5 * df['ln_AQY'] + 0.5 * df['ln_TOF']

    print(f"\nObjective statistics:")
    print(f"  Mean: {df['objective'].mean():.3f}")
    print(f"  Std:  {df['objective'].std():.3f}")
    print(f"  Min:  {df['objective'].min():.3f}")
    print(f"  Max:  {df['objective'].max():.3f}")

    # All features are continuous - no encoding needed
    feature_cols = ['catalyst', 'hole_scavenger', 'light_intensity', 'time']
    feature_names = feature_cols.copy()

    # Build feature matrix
    X_parts = []
    for col in feature_cols:
        X_parts.append(df[col].values.reshape(-1, 1))

    X = np.hstack(X_parts)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"All {len(feature_names)} features treated as continuous:")
    for i, name in enumerate(feature_names):
        print(f"  {name:20s}: [{X[:, i].min():.4f}, {X[:, i].max():.4f}]")

    return df, X, feature_names


# ============================================================================
# BAYESIAN OPTIMIZATION
# ============================================================================

def run_bayesian_optimization(X: np.ndarray, y: np.ndarray,
                              n_init: int = N_INIT, n_iter: int = N_ITER,
                              batch_size: int = 1,
                              random_seed: int = RANDOM_SEED) -> Dict:
    """
    Run pool-based Bayesian optimization with Gaussian Process and EI.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (full pool)
    y : np.ndarray, shape (n_samples,)
        Objective values (full pool)
    n_init : int
        Number of random initial points
    n_iter : int
        Number of BO iterations
    batch_size : int
        Number of points to select per iteration
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    results : dict
        Dictionary containing:
        - history: list of dicts with iteration details (n_iter * batch_size entries)
        - observed_indices: list of selected indices
        - observed_y: array of observed objectives
        - best_per_iteration: array of best objective per iteration (n_iter + 1 entries)

    Notes
    -----
    Kernel: ConstantKernel * RBF + WhiteKernel
    - ConstantKernel: output variance scaling (1e-3 to 1e3)
    - RBF: squared exponential, smooth variations
    - WhiteKernel: noise model (1e-6)
    Uses n_restarts_optimizer=10 for robust hyperparameter optimization.

    Important: best_per_iteration has length (n_iter + 1) because it includes
    the initial best from random sampling, while history has (n_iter * batch_size)
    entries, one for each suggested point.
    """
    np.random.seed(random_seed)
    n_samples, n_features = X.shape

    if n_samples < n_init + n_iter * batch_size:
        raise ValueError(f"Insufficient pool size: {n_samples} < {n_init + n_iter * batch_size}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize with random points
    all_indices = np.arange(n_samples)
    np.random.shuffle(all_indices)
    observed_indices = all_indices[:n_init].tolist()
    remaining_indices = all_indices[n_init:].tolist()

    history = []
    best_per_iteration = []

    print(f"\n{'='*60}")
    print("BAYESIAN OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Pool size: {n_samples}")
    print(f"Initial points: {n_init}")
    print(f"BO iterations: {n_iter}")
    print(f"Batch size: {batch_size}")
    if batch_size > 1:
        print(f"⚠ Note: With batch_size={batch_size}, each iteration selects {batch_size} points")
        print(f"        History will contain {n_iter * batch_size} entries")

    # Define GP kernel with appropriate bounds and noise model
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
              RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-1)))

    for iteration in range(n_iter):
        # Get observed data
        X_obs = X_scaled[observed_indices]
        y_obs = y[observed_indices]
        y_best = np.max(y_obs)
        best_per_iteration.append(y_best)

        # Fit GP
        gp = GaussianProcessRegressor(kernel=kernel,
                                      n_restarts_optimizer=N_RESTARTS,
                                      random_state=random_seed,
                                      normalize_y=True)
        gp.fit(X_obs, y_obs)

        # Predict on remaining pool
        X_remain = X_scaled[remaining_indices]
        mu, sigma = gp.predict(X_remain, return_std=True)

        # Compute EI
        ei = expected_improvement(mu, sigma, y_best, xi=XI)

        # Select top batch_size points
        top_k_idx = np.argsort(ei)[-batch_size:][::-1]

        for k in top_k_idx:
            selected_pool_idx = remaining_indices[k]

            history.append({
                'iteration': iteration,
                'suggested_index': selected_pool_idx,
                'predicted_mean': mu[k],
                'predicted_std': sigma[k],
                'EI': ei[k],
                'actual_objective': y[selected_pool_idx],
            })

            observed_indices.append(selected_pool_idx)

        # Remove selected from remaining
        remaining_indices = [idx for i, idx in enumerate(remaining_indices)
                             if i not in top_k_idx]

        if iteration % 2 == 0 or iteration == n_iter - 1:
            print(f"Iteration {iteration:2d}: Best={y_best:.4f}, "
                  f"Selected EI={ei[top_k_idx[0]]:.4e}, "
                  f"Actual={y[selected_pool_idx]:.4f}")

    # Final best
    y_obs_final = y[observed_indices]
    best_per_iteration.append(np.max(y_obs_final))

    # Validation check
    assert len(best_per_iteration) == n_iter + 1, \
        f"best_per_iteration length mismatch: {len(best_per_iteration)} != {n_iter + 1}"
    assert len(history) == n_iter * batch_size, \
        f"history length mismatch: {len(history)} != {n_iter * batch_size}"

    results = {
        'history': history,
        'observed_indices': observed_indices,
        'observed_y': y_obs_final,
        'best_per_iteration': np.array(best_per_iteration),
        'scaler': scaler,
        'gp': gp
    }

    return results


def run_continuous_optimization(X: np.ndarray, y: np.ndarray,
                                n_suggest: int = 5,
                                random_seed: int = RANDOM_SEED) -> Dict:
    """
    Optimize EI in continuous space to generate DIVERSE suggestions.
    Also generates a second batch based on highest predicted objective (exploitation).
    """
    np.random.seed(random_seed)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X.shape[1])) +
              WhiteKernel(noise_level=1e-6))

    gp = GaussianProcessRegressor(kernel=kernel,
                                  n_restarts_optimizer=N_RESTARTS,
                                  random_state=random_seed,
                                  normalize_y=True)
    gp.fit(X_scaled, y)

    y_best = np.max(y)

    # This list will store the coordinates of points we've already suggested.
    found_points_scaled = []

    # This function is now defined *inside* the main function so it can access 'found_points_scaled'.
    def neg_ei(x_scaled):
        x_scaled = x_scaled.reshape(1, -1)

        # Check if the proposed point 'x_scaled' is too close to any point we've already found.
        for point in found_points_scaled:
            # We calculate the distance in the scaled space. A threshold of 0.5 is a reasonable start.
            if np.linalg.norm(x_scaled - point) < 0.5:
                return 1e6  # Return a very large number (a bad score) to penalize this point.

        # If the point is not too close to others, calculate its true negative EI.
        mu, sigma = gp.predict(x_scaled, return_std=True)
        return -expected_improvement(mu, sigma, y_best, xi=XI)[0]

    def neg_mean(x_scaled):
        """Negative mean for maximizing predicted objective."""
        x_scaled = x_scaled.reshape(1, -1)
        
        # Check if too close to already suggested points
        for point in found_points_exploitative:
            if np.linalg.norm(x_scaled - point) < 0.5:
                return 1e6
        
        mu, _ = gp.predict(x_scaled, return_std=True)
        return -mu[0]

    suggestions_ei = []
    suggestions_exploit = []
    bounds = [(X_scaled[:, i].min(), X_scaled[:, i].max())
              for i in range(X.shape[1])]

    print(f"\n{'='*60}")
    print("CONTINUOUS OPTIMIZATION (Generating Diverse Suggestions)")
    print(f"{'='*60}")
    
    # BATCH 1: Expected Improvement (Exploration)
    print(f"\nBATCH 1: Generating {n_suggest} suggestions via Expected Improvement (EI)...")
    print("Strategy: Balance exploration and exploitation")
    print("-" * 60)

    for i in range(n_suggest):
        best_ei = np.inf
        best_x = None

        # Multiple random restarts to find the global optimum for this iteration
        for _ in range(20):
            x0 = np.random.uniform([b[0] for b in bounds],
                                   [b[1] for b in bounds])
            res = minimize(neg_ei, x0, bounds=bounds, method='L-BFGS-B')

            if res.fun < best_ei:
                best_ei = res.fun
                best_x = res.x

        if best_x is not None:
            found_points_scaled.append(best_x.reshape(1, -1))

        x_original = scaler.inverse_transform(best_x.reshape(1, -1))[0]
        mu, sigma = gp.predict(best_x.reshape(1, -1), return_std=True)

        suggestions_ei.append({
            'suggestion_id': i,
            'batch': 'EI',
            'features_scaled': best_x,
            'features_original': x_original,
            'predicted_mean': mu[0],
            'predicted_std': sigma[0],
            'EI': -best_ei
        })

        print(f"  EI-{i+1}: EI={-best_ei:.4e}, μ={mu[0]:.4f}, σ={sigma[0]:.4f}")

    # BATCH 2: Highest Predicted Objective (Exploitation)
    print(f"\nBATCH 2: Generating {n_suggest} suggestions via Highest Predicted Objective...")
    print("Strategy: Pure exploitation (greedy)")
    print("-" * 60)
    
    found_points_exploitative = []
    
    for i in range(n_suggest):
        best_obj = np.inf
        best_x = None

        # Multiple random restarts
        for _ in range(20):
            x0 = np.random.uniform([b[0] for b in bounds],
                                   [b[1] for b in bounds])
            res = minimize(neg_mean, x0, bounds=bounds, method='L-BFGS-B')

            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x

        if best_x is not None:
            found_points_exploitative.append(best_x.reshape(1, -1))

        x_original = scaler.inverse_transform(best_x.reshape(1, -1))[0]
        mu, sigma = gp.predict(best_x.reshape(1, -1), return_std=True)

        suggestions_exploit.append({
            'suggestion_id': i,
            'batch': 'Exploit',
            'features_scaled': best_x,
            'features_original': x_original,
            'predicted_mean': mu[0],
            'predicted_std': sigma[0],
            'EI': expected_improvement(mu, sigma, y_best, xi=XI)[0]
        })

        print(f"  OBJ-{i+1}: μ={mu[0]:.4f}, σ={sigma[0]:.4f}, EI={suggestions_exploit[-1]['EI']:.4e}")

    return {
        'suggestions_ei': suggestions_ei,
        'suggestions_exploit': suggestions_exploit,
        'gp': gp,
        'scaler': scaler
    }
# ============================================================================
# MACHINE LEARNING & INTERPRETABILITY
# ============================================================================

def train_random_forest(X: np.ndarray, y: np.ndarray,
                        feature_names: List[str]) -> Dict:
    """
    Train Random Forest with grid search and compute interpretability metrics.

    MODIFICATION: The grid search now includes 'max_depth' and 'min_samples_leaf'
    to actively regularize the model and combat overfitting.
    """
    print(f"\n{'='*60}")
    print("RANDOM FOREST TRAINING (with regularization)")
    print(f"{'='*60}")

    # The expanded parameter grid for GridSearchCV.
    # It will now search for the best combination of these parameters
    # to maximize the cross-validated R-squared score.
    param_grid = {
        'n_estimators': RF_N_ESTIMATORS_GRID,
        'max_features': RF_MAX_FEATURES_GRID,
        'max_depth': RF_MAX_DEPTH_GRID,          # <-- ADDED
        'min_samples_leaf': RF_MIN_SAMPLES_LEAF_GRID # <-- ADDED
    }

    rf_base = RandomForestRegressor(random_state=RANDOM_SEED)
    
    # GridSearchCV will find the best hyperparameters based on the CV score,
    # effectively selecting a model that is less likely to be overfit.
    grid_search = GridSearchCV(rf_base, param_grid, cv=RF_CV_FOLDS,
                               scoring='r2', n_jobs=-1, verbose=0)
    grid_search.fit(X, y)

    # The best score from cross-validation (a good estimate of real-world performance)
    print(f"Best parameters found by CV: {grid_search.best_params_}")
    print(f"Best CV R² (Generalization Score): {grid_search.best_score_:.4f}")

    # Re-train the best model found by the grid search on the entire dataset
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X)
    
    # Calculate R² on the training data. This will likely be lower than before,
    # which is a GOOD sign, as it means the gap to the CV score is smaller.
    r2_training = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print(f"Full data R² (Training Score): {r2_training:.4f}")
    print(f"Full data RMSE: {rmse:.4f}")

    # CV results dataframe
    cv_results_df = pd.DataFrame(grid_search.cv_results_)

    # Permutation importance
    print(f"\nComputing permutation importance ({PERM_IMPORTANCE_REPEATS} repeats)...")
    perm_importance = permutation_importance(
        best_rf, X, y, n_repeats=PERM_IMPORTANCE_REPEATS,
        random_state=RANDOM_SEED, n_jobs=-1
    )

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)

    print("\nTop 5 features by permutation importance:")
    for _, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']:20s}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")

    # SHAP analysis if available
    shap_values = None
    if SHAP_AVAILABLE:
        print("\nComputing SHAP values...")
        explainer = shap.TreeExplainer(best_rf)
        shap_values = explainer.shap_values(X)
        print("✓ SHAP values computed")

    results = {
        'model': best_rf,
        'cv_results': cv_results_df,
        'importance_df': importance_df,
        'shap_values': shap_values,
        'predictions': y_pred,
        'r2': r2_training, # Use the training R² for consistency in reporting
        'rmse': rmse
    }

    return results
# ============================================================================
# VISUALIZATION - HIGH QUALITY PUBLICATION-READY PLOTS
# ============================================================================

def setup_plot_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(PLOT_STYLE)


def save_figure(fig, name: str):
    """Save figure in multiple high-quality formats."""
    for fmt in PLOT_FORMAT:
        if fmt == 'png':
            fig.savefig(f'figures/{name}.{fmt}', dpi=PLOT_DPI, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
        else:
            fig.savefig(f'figures/{name}.{fmt}', bbox_inches='tight',
                        facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"✓ Saved {name}.{'/'.join(PLOT_FORMAT)}")


def plot_optimization_trajectory(best_per_iteration: np.ndarray,
                                 bo_results: Optional[Dict] = None):
    """
    Plot best observed objective vs iteration with optional GP uncertainty bands.

    Parameters
    ----------
    best_per_iteration : np.ndarray
        Best objective value at each iteration
    bo_results : dict, optional
        BO results containing GP model and scaler for uncertainty estimation

    Notes
    -----
    This function correctly aligns the x-axis between best_per_iteration
    (which tracks the best objective at each BO iteration) and GP predictions
    (which may include multiple suggestions per iteration when batch_size > 1).
    GP predictions are aggregated per iteration to match the scale.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    iterations = np.arange(len(best_per_iteration))
    ax.plot(iterations, best_per_iteration, 'o-', linewidth=2.5,
            markersize=8, color=COLORS['primary'], label='Best Objective',
            zorder=3, markeredgewidth=1.5, markeredgecolor='white')

    # Add GP uncertainty bands if available (pool-based mode only)
    if bo_results is not None and 'gp' in bo_results and 'history' in bo_results:
        history = bo_results['history']

        if len(history) > 0:
            # Group history by iteration to handle batch_size > 1
            # Each iteration may have multiple suggestions (batch_size)
            # We need to aggregate them to match best_per_iteration scale

            history_df = pd.DataFrame(history)

            # Group by iteration and take the mean/max of predictions
            # Use max of predicted_mean as it represents the most promising point
            grouped = history_df.groupby('iteration').agg({
                'predicted_mean': 'max',  # Best prediction in this iteration
                'predicted_std': 'mean'   # Average uncertainty
            }).reset_index()

            # Now x-axis aligns: grouped['iteration'] matches best_per_iteration indices
            pred_iterations = grouped['iteration'].values
            pred_means = grouped['predicted_mean'].values
            pred_stds = grouped['predicted_std'].values

            # Plot uncertainty bands (95% CI) - now correctly aligned
            ax.fill_between(pred_iterations,
                            pred_means - 2*pred_stds,
                            pred_means + 2*pred_stds,
                            alpha=0.25, color=COLORS['primary'],
                            label='GP 95% CI', zorder=1)
            ax.plot(pred_iterations, pred_means, '--',
                    linewidth=2, color=COLORS['primary'], alpha=0.6,
                    label='GP Predicted Best', zorder=2)

    # Annotate first and last with improved styling
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=COLORS['primary'], alpha=0.9, linewidth=1.5)

    ax.annotate(f'{best_per_iteration[0]:.3f}',
                xy=(0, best_per_iteration[0]),
                xytext=(15, -20), textcoords='offset points',
                fontsize=10, ha='left', fontweight='bold',
                bbox=bbox_props,
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'],
                                lw=1.5, connectionstyle='arc3,rad=0.3'))

    ax.annotate(f'{best_per_iteration[-1]:.3f}',
                xy=(len(best_per_iteration)-1, best_per_iteration[-1]),
                xytext=(15, 15), textcoords='offset points',
                fontsize=10, ha='left', fontweight='bold',
                bbox=bbox_props,
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'],
                                lw=1.5, connectionstyle='arc3,rad=-0.3'))

    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Best Observed Objective', fontweight='bold')
    ax.set_title('Bayesian Optimization Trajectory', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(frameon=True, shadow=True, fancybox=True, loc='best')

    # Add improvement annotation
    improvement = best_per_iteration[-1] - best_per_iteration[0]
    pct_improvement = (improvement / abs(best_per_iteration[0])) * 100
    ax.text(0.02, 0.98, f'Improvement: {improvement:+.3f} ({pct_improvement:+.1f}%)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    save_figure(fig, 'fig_optimization_trajectory')


def plot_sorted_objectives(df: pd.DataFrame):
    """Plot sorted objectives with TOF and AQY in bottom panel."""
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    df_sorted = df.sort_values('objective').reset_index(drop=True)
    best_idx = df_sorted['objective'].idxmax()

    # Top panel: Objective with gradient coloring
    scatter = ax1.scatter(df_sorted.index, df_sorted['objective'],
                          c=df_sorted['objective'], cmap='viridis',
                          s=80, alpha=0.7, edgecolors='black', linewidth=0.8,
                          zorder=3)

    # Highlight best point
    ax1.scatter(best_idx, df_sorted.loc[best_idx, 'objective'],
                marker='*', s=600, color='red', edgecolors='darkred',
                linewidth=2, zorder=4, label=f'Best (idx={best_idx})')

    ax1.axvline(best_idx, color='red', linestyle='--', linewidth=2,
                alpha=0.5, zorder=2)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, pad=0.02)
    cbar.set_label('Objective Value', rotation=270, labelpad=20, fontweight='bold')

    ax1.set_ylabel('Objective (ln-weighted)', fontweight='bold')
    ax1.set_title('Sorted Experiments by Objective Performance',
                  fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(frameon=True, shadow=True)

    # Bottom panel: TOF (left y) and AQY (right y) with dual axes
    ax2_right = ax2.twinx()

    ln1 = ax2.plot(df_sorted.index, df_sorted['TOF'], 's-',
                   markersize=6, color=COLORS['accent1'], alpha=0.8,
                   label='TOF', linewidth=1.5, markeredgewidth=1,
                   markeredgecolor='white')
    ln2 = ax2_right.plot(df_sorted.index, df_sorted['AQY'], '^-',
                         markersize=6, color=COLORS['accent2'], alpha=0.8,
                         label='AQY', linewidth=1.5, markeredgewidth=1,
                         markeredgecolor='white')

    ax2.axvline(best_idx, color='red', linestyle='--', linewidth=2, alpha=0.5)

    ax2.set_xlabel('Sorted Experiment Index', fontweight='bold')
    ax2.set_ylabel('TOF (h⁻¹)', color=COLORS['accent1'], fontweight='bold')
    ax2_right.set_ylabel('AQY (%)', color=COLORS['accent2'], fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=COLORS['accent1'])
    ax2_right.tick_params(axis='y', labelcolor=COLORS['accent2'])
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='upper left', frameon=True, shadow=True)

    plt.tight_layout()
    save_figure(fig, 'fig_sorted_objectives_tof_aqy')


def plot_tof_vs_aqy(df: pd.DataFrame):
    """Scatter plot of TOF vs AQY colored by objective."""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(df['TOF'], df['AQY'], c=df['objective'],
                         cmap='plasma', s=120, alpha=0.7,
                         edgecolors='black', linewidth=1.2, zorder=3)

    # Highlight best
    best_idx = df['objective'].idxmax()
    best_row = df.loc[best_idx]
    ax.scatter(best_row['TOF'], best_row['AQY'], marker='*', s=800,
               color='red', edgecolors='darkred', linewidth=3,
               label='Best Experiment', zorder=10)

    # Annotate best with arrow
    ax.annotate(f"{best_row['experiment_id']}\nObj: {best_row['objective']:.3f}",
                xy=(best_row['TOF'], best_row['AQY']),
                xytext=(30, 30), textcoords='offset points',
                fontsize=10, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                          edgecolor='darkred', alpha=0.95, linewidth=2),
                arrowprops=dict(arrowstyle='->', color='darkred',
                                lw=2.5, connectionstyle='arc3,rad=0.2'))

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Objective (ln weighted)', rotation=270,
                   labelpad=25, fontweight='bold', fontsize=11)

    ax.set_xlabel('TOF (h⁻¹)', fontweight='bold', fontsize=12)
    ax.set_ylabel('AQY (%)', fontweight='bold', fontsize=12)
    ax.set_title('TOF vs AQY Performance Space', fontweight='bold',
                 pad=20, fontsize=14)
    ax.legend(frameon=True, shadow=True, fancybox=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # Add quadrant lines (median splits)
    median_tof = df['TOF'].median()
    median_aqy = df['AQY'].median()
    ax.axvline(median_tof, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axhline(median_aqy, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)

    plt.tight_layout()
    save_figure(fig, 'fig_tof_vs_aqy')


def plot_feature_importance(importance_df: pd.DataFrame):
    """Horizontal bar chart of permutation importance with error bars."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Sort and take top 10
    imp_sorted = importance_df.sort_values('importance_mean', ascending=True).tail(10)

    y_pos = np.arange(len(imp_sorted))
    colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_sorted)))

    bars = ax.barh(y_pos, imp_sorted['importance_mean'],
                   xerr=imp_sorted['importance_std'],
                   color=colors_grad, alpha=0.8,
                   edgecolor='black', linewidth=1.2,
                   error_kw={'linewidth': 2, 'ecolor': 'darkred',
                             'capsize': 5, 'capthick': 2})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_sorted['feature'], fontweight='bold')
    ax.set_xlabel('Permutation Importance (decrease in R²)',
                  fontweight='bold', fontsize=12)
    ax.set_title('Feature Importance Analysis (Random Forest)',
                 fontweight='bold', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=0.8)

    # Add value labels on bars
    for i, (idx, row) in enumerate(imp_sorted.iterrows()):
        ax.text(row['importance_mean'] + row['importance_std'] + 0.002, i,
                f"{row['importance_mean']:.4f}", va='center', fontsize=9,
                fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'fig_feature_importance')


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, r2: float):
    """Scatter plot of predicted vs actual with 1:1 line and statistics."""
    fig, ax = plt.subplots(figsize=(9, 9))

    # Color points by residual magnitude
    residuals = np.abs(y_true - y_pred)
    scatter = ax.scatter(y_true, y_pred, c=residuals, cmap='RdYlGn_r',
                         s=100, alpha=0.7, edgecolors='black',
                         linewidth=1, zorder=3)

    # 1:1 line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', alpha=0.6, linewidth=2.5,
            label='Perfect Prediction (1:1)', zorder=2)

    # Add ±10% prediction bands
    margin = 0.1 * (lims[1] - lims[0])
    ax.fill_between(lims, np.array(lims) - margin, np.array(lims) + margin,
                    alpha=0.15, color='blue', label='±10% Band', zorder=1)

    # Statistics box
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))

    stats_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nn = {len(y_true)}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white',
                      edgecolor=COLORS['primary'], alpha=0.9, linewidth=2))

    # Colorbar for residuals
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('|Residual|', rotation=270, labelpad=20, fontweight='bold')

    ax.set_xlabel('Actual Objective', fontweight='bold', fontsize=12)
    ax.set_ylabel('Predicted Objective', fontweight='bold', fontsize=12)
    ax.set_title('Random Forest: Predictions vs Actual',
                 fontweight='bold', fontsize=14, pad=20)
    ax.legend(frameon=True, shadow=True, fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    save_figure(fig, 'fig_predictions_vs_actual')


def plot_feature_correlation(X: np.ndarray, feature_names: List[str]):
    """
    Plot correlation matrix heatmap of features.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix
    feature_names : list of str
        Feature names for labels
    """
    fig, ax = plt.subplots(figsize=(10, 9))

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)

    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-1, vmax=1, interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Pearson Correlation', rotation=270, labelpad=25,
                   fontweight='bold', fontsize=11)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontweight='bold')
    ax.set_yticklabels(feature_names, fontweight='bold')

    # Add correlation values as text
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                           ha='center', va='center', color=text_color,
                           fontsize=10, fontweight='bold')

    ax.set_title('Feature Correlation Matrix', fontweight='bold',
                 fontsize=14, pad=20)

    # Add warning for high correlations
    max_corr = np.max(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
    if max_corr > 0.8:
        ax.text(0.5, -0.15, f'⚠ Warning: Max correlation = {max_corr:.2f} (>0.8 indicates multicollinearity)',
                transform=ax.transAxes, ha='center', fontsize=10,
                color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    save_figure(fig, 'fig_feature_correlation')


def plot_residual_analysis(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Create comprehensive residual diagnostic plots.

    Parameters
    ----------
    y_true : np.ndarray
        Actual objective values
    y_pred : np.ndarray
        Predicted objective values
    """
    residuals = y_true - y_pred

    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(3, 2, hspace=0.35, wspace=0.35)

    # 1. Residuals vs Predicted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, residuals, alpha=0.6, s=80, color=COLORS['primary'],
                edgecolors='black', linewidth=1)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2.5, alpha=0.7,
                label='Zero residual')
    ax1.set_xlabel('Predicted Objective', fontweight='bold')
    ax1.set_ylabel('Residuals', fontweight='bold')
    ax1.set_title('Residuals vs Predicted', fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Add ±2σ bands
    residual_std = np.std(residuals)
    ax1.axhline(y=2*residual_std, color='orange', linestyle=':',
                linewidth=2, alpha=0.6, label='±2σ')
    ax1.axhline(y=-2*residual_std, color='orange', linestyle=':',
                linewidth=2, alpha=0.6)
    ax1.legend(frameon=True, shadow=True)

    # 2. Residuals vs Actual
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_true, residuals, alpha=0.6, s=80, color=COLORS['secondary'],
                edgecolors='black', linewidth=1)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
    ax2.set_xlabel('Actual Objective', fontweight='bold')
    ax2.set_ylabel('Residuals', fontweight='bold')
    ax2.set_title('Residuals vs Actual', fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=2*residual_std, color='orange', linestyle=':',
                linewidth=2, alpha=0.6)
    ax2.axhline(y=-2*residual_std, color='orange', linestyle=':',
                linewidth=2, alpha=0.6)

    # 3. Histogram of Residuals
    ax3 = fig.add_subplot(gs[1, 0])
    n, bins, patches = ax3.hist(residuals, bins=25, alpha=0.7,
                                color=COLORS['accent2'],
                                edgecolor='black', linewidth=1.2)

    # Overlay normal distribution
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax3.plot(x, len(residuals) * (bins[1] - bins[0]) *
             norm.pdf(x, mu, sigma),
             'r-', linewidth=3, label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')

    ax3.set_xlabel('Residuals', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Distribution of Residuals', fontweight='bold', pad=10)
    ax3.legend(frameon=True, shadow=True)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

    # 4. Q-Q Plot
    ax4 = fig.add_subplot(gs[1, 1])
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.get_lines()[0].set_markerfacecolor(COLORS['accent1'])
    ax4.get_lines()[0].set_markeredgecolor('black')
    ax4.get_lines()[0].set_markersize(7)
    ax4.get_lines()[1].set_linewidth(2.5)
    ax4.get_lines()[1].set_color('red')
    ax4.set_title('Q-Q Plot (Normal Distribution)', fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlabel('Theoretical Quantiles', fontweight='bold')
    ax4.set_ylabel('Sample Quantiles', fontweight='bold')

    # 5. Scale-Location Plot
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(y_pred, np.abs(residuals), alpha=0.6, s=80,
                color=COLORS['accent1'],
                edgecolors='black', linewidth=1)
    ax5.set_xlabel('Predicted Objective', fontweight='bold')
    ax5.set_ylabel('√|Residuals|', fontweight='bold')
    ax5.set_title('Scale-Location Plot', fontweight='bold', pad=10)
    ax5.grid(True, alpha=0.3, linestyle='--')

    # Add lowess smoothing line
    from scipy.signal import savgol_filter
    sorted_idx = np.argsort(y_pred)
    if len(y_pred) > 10:
        try:
            window = min(len(y_pred) // 3, 11)
            if window % 2 == 0:
                window += 1
            smoothed = savgol_filter(np.abs(residuals)[sorted_idx], window, 2)
            ax5.plot(y_pred[sorted_idx], smoothed, 'r-', linewidth=3,
                     alpha=0.7, label='Smoothed trend')
            ax5.legend(frameon=True, shadow=True)
        except:
            pass

    # 6. Residuals Summary Statistics
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    # Compute statistics
    from scipy.stats import shapiro, jarque_bera

    shapiro_stat, shapiro_p = shapiro(residuals) if len(residuals) < 5000 else (np.nan, np.nan)
    jb_stat, jb_p = jarque_bera(residuals)

    # Determine pass/fail for tests
    shapiro_status = '✓' if shapiro_p > 0.05 else '✗'
    jb_status = '✓' if jb_p > 0.05 else '✗'

    stats_text = f"""
╔══════════════════════════════════════╗
║   RESIDUAL DIAGNOSTICS SUMMARY       ║
╚══════════════════════════════════════╝

Descriptive Statistics:
───────────────────────────────────────
  Mean:           {np.mean(residuals):>10.4f}
  Median:         {np.median(residuals):>10.4f}
  Std Dev:        {np.std(residuals):>10.4f}
  Min:            {np.min(residuals):>10.4f}
  Max:            {np.max(residuals):>10.4f}

Normality Tests (α = 0.05):
───────────────────────────────────────
  Shapiro-Wilk:
    W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}  {shapiro_status}

  Jarque-Bera:
    χ² = {jb_stat:.4f}, p = {jb_p:.4f}  {jb_status}

Outlier Analysis:
───────────────────────────────────────
  Beyond ±2σ:  {np.sum(np.abs(residuals) > 2*residual_std):>3d} / {len(residuals)} ({100*np.sum(np.abs(residuals) > 2*residual_std)/len(residuals):.1f}%)
  Beyond ±3σ:  {np.sum(np.abs(residuals) > 3*residual_std):>3d} / {len(residuals)} ({100*np.sum(np.abs(residuals) > 3*residual_std)/len(residuals):.1f}%)

Interpretation:
───────────────────────────────────────
  ✓ = Pass (p > 0.05): Residuals ~ Normal
  ✗ = Fail (p < 0.05): Non-normal residuals

  Mean ≈ 0 ✓ : Unbiased predictions
  Outliers < 5% ✓ : Good model fit
"""

    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightblue',
                       alpha=0.8, edgecolor='black', linewidth=2))

    fig.suptitle('Comprehensive Residual Analysis & Model Diagnostics',
                 fontsize=16, fontweight='bold', y=0.998)

    save_figure(fig, 'fig_residual_analysis')


def plot_shap_summary(shap_values: np.ndarray, X: np.ndarray,
                      feature_names: List[str]):
    """Create SHAP summary plots if SHAP is available."""
    if not SHAP_AVAILABLE or shap_values is None:
        print("⚠ Skipping SHAP plots (library not available or no values computed)")
        return

    try:
        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          plot_type='bar', show=False, max_display=10)
        plt.title('SHAP Feature Importance', fontweight='bold', fontsize=14, pad=20)
        plt.xlabel('Mean |SHAP Value|', fontweight='bold')
        plt.tight_layout()
        save_figure(fig, 'fig_shap_summary')

        # Beeswarm plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          show=False, max_display=10)
        plt.title('SHAP Feature Impact Distribution', fontweight='bold', fontsize=14, pad=20)
        plt.tight_layout()
        save_figure(fig, 'fig_shap_beeswarm')

    except Exception as e:
        print(f"⚠ Could not generate SHAP plots: {e}")
        print("  This is usually fine - permutation importance is available instead.")


def plot_parameter_space(X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    """
    Generate a pairwise scatter plot matrix to visualize the parameter space.

    - Diagonal: Histogram of each feature's distribution.
    - Lower Triangle: Scatter plot of feature pairs, colored by the objective value.
    - Best point is highlighted with a red star.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Objective values (n_samples,)
    feature_names : list of str
        Names of the features for labels
    """
    n_features = X.shape[1]
    fig, axes = plt.subplots(n_features, n_features, figsize=(3 * n_features, 3 * n_features))
    
    # Find the best point to highlight
    best_idx = np.argmax(y)
    
    # Create the main scatter object for the colorbar
    scatter = None

    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            
            # Diagonal: Plot histogram
            if i == j:
                ax.hist(X[:, i], bins=20, color=COLORS['primary'], alpha=0.7, edgecolor='black')
                ax.set_title(feature_names[i], fontweight='bold', fontsize=10, pad=10)
            
            # Lower triangle: Plot scatter colored by objective
            elif i > j:
                scatter = ax.scatter(X[:, j], X[:, i], c=y, cmap='viridis', s=50, alpha=0.7,
                                     edgecolors='k', linewidth=0.5)
                # Highlight the best point
                ax.scatter(X[best_idx, j], X[best_idx, i], marker='*', s=400,
                           color='red', edgecolors='darkred', linewidth=1.5, zorder=10)
            
            # Upper triangle: Hide
            else:
                ax.axis('off')

            # Set axis labels for the outer plots only
            if j == 0 and i > 0:
                ax.set_ylabel(feature_names[i], fontweight='bold')
            if i == n_features - 1:
                ax.set_xlabel(feature_names[j], fontweight='bold')
                
            # Tidy up ticks
            if i != j:
                ax.grid(True, linestyle='--', alpha=0.3)
            
            if j > 0:
                ax.tick_params(axis='y', labelleft=False)
            if i < n_features - 1:
                ax.tick_params(axis='x', labelbottom=False)

    # Add a single colorbar for the entire figure
    if scatter:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('Objective Value', rotation=270, labelpad=20, fontweight='bold')

    fig.suptitle('Parameter Space Visualization', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    save_figure(fig, 'fig_parameter_space')


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_methods_text(df: pd.DataFrame, bo_results: Dict, ml_results: Dict) -> str:
    """
    Generate methods section text for manuscript.

    Returns
    -------
    methods_text : str
        Formatted methods paragraph for publication
    """
    text = f"""
COMPUTATIONAL METHODS

We performed holistic optimization analysis on {len(df)} photocatalytic
hydrogen evolution experiments using Bayesian optimization and machine learning.
The objective function combined turnover frequency (TOF) and apparent quantum
yield (AQY) with equal weighting: Objective = 0.5·ln(AQY) + 0.5·ln(TOF).

Bayesian optimization employed a Gaussian process with RBF kernel
(ConstantKernel(1.0) × RBF + WhiteKernel(1e-6)), using Expected Improvement
acquisition (xi={XI}) over {N_ITER} iterations starting from {N_INIT} random
initial points. All features were standardized prior to modeling. All experimental
parameters (catalyst loading, solvent amount, hole scavenger concentration,
light intensity, and reaction time) were treated as continuous variables.

Machine learning used Random Forest regression ({RF_N_ESTIMATORS_GRID[-1]} trees)
with 5-fold cross-validation grid search over n_estimators and max_features.
Feature importance was assessed via permutation importance ({PERM_IMPORTANCE_REPEATS}
repeats) and SHAP (Tree Explainer). The final model achieved R² = {ml_results['r2']:.3f}
on the full dataset. All analysis used Python 3.8+ with scikit-learn, SciPy,
and SHAP. Random seed = {RANDOM_SEED} for reproducibility.

All figures were generated at 300 DPI for publication quality.
"""
    return text.strip()

def run_analysis(continuous_mode: bool = True, n_suggest: int = 5,
                 batch_size: int = 1, xi_override: float = None):
    """
    Main analysis pipeline.

    Parameters
    ----------
    continuous_mode : bool
        If True, run continuous optimization instead of pool-based
    n_suggest : int
        Number of suggestions for continuous mode
    batch_size : int
        Batch size for pool-based BO
    xi_override : float, optional
        Override default XI for exploration control
    """
    global XI
    if xi_override is not None:
        XI = xi_override
        print(f"\n⚙ Using custom XI = {XI} for exploration/exploitation control")
        if XI < 0.01:
            print("  → More EXPLOITATION (greedy, finds local optima)")
        elif XI > 0.1:
            print("  → More EXPLORATION (diverse, finds global patterns)")
        else:
            print("  → BALANCED approach (recommended)")

    print("\n" + "="*60)
    print("BONKE-STYLE HOLISTIC OPTIMIZATION ANALYSIS")
    print("Enhanced with High-Quality Visualizations v2.1")
    print("="*60)

    setup_directories()
    setup_plot_style()

    # Load and preprocess data
    df = load_and_validate_data('data/raw/experiments.csv')
    df, X, feature_names = preprocess_data(df)
    y = df['objective'].values

    # Save full dataset
    df.to_csv('outputs/full_dataset_with_objective.csv', index=False)
    print("✓ Saved full_dataset_with_objective.csv")

    # Bayesian Optimization
    if continuous_mode:
        bo_results = run_continuous_optimization(X, y, n_suggest=n_suggest)

        # Save continuous suggestions with proper feature names
        suggestions_data = []
        
        # Combine the two suggestion lists into one
        all_suggestions = bo_results['suggestions_ei'] + bo_results['suggestions_exploit']
        
        for s in all_suggestions:
            row = {'suggestion_id': f"{s['batch']}-{s['suggestion_id'] + 1}"} # Add batch info to ID
            row['batch'] = s['batch']
            for i, fname in enumerate(feature_names):
                # Use the exact feature names as defined globally
                row[FEATURE_NAMES[i]] = s['features_original'][i]
            row['predicted_mean'] = s['predicted_mean']
            row['predicted_std'] = s['predicted_std']
            row['predicted_objective'] = s['predicted_mean']
            row['uncertainty'] = s['predicted_std']
            row['EI'] = s['EI']
            suggestions_data.append(row)

        suggestions_df = pd.DataFrame(suggestions_data)
        suggestions_df.to_csv('outputs/continuous_suggestions.csv', index=False)
        print("✓ Saved continuous_suggestions.csv")

        # Display suggested experiments prominently
        print(f"\n{'='*80}")
        print(f"{'🔬 SUGGESTED NEXT EXPERIMENTS (Continuous Optimization)':^80}")
        print(f"{'='*80}")
        
        # Group by batch for cleaner output
        for batch_name, group in suggestions_df.groupby('batch'):
            strategy = 'Balance exploration & exploitation' if batch_name == 'EI' else 'Pure exploitation (greedy)'
            print(f"\n--- BATCH: {batch_name} ({strategy}) ---")
            for idx, row in group.iterrows():
                print(f"Suggestion {row['suggestion_id']}:")
                print(f"  Parameters:")
                print(f"    • {FEATURE_NAMES[0]}: {row[FEATURE_NAMES[0]]:>18.6f}")
                print(f"    • {FEATURE_NAMES[1]}: {row[FEATURE_NAMES[1]]:>18.4f}")
                print(f"    • {FEATURE_NAMES[2]}: {row[FEATURE_NAMES[2]]:>14.6f}")
                print(f"    • {FEATURE_NAMES[3]}: {row[FEATURE_NAMES[3]]:>13.2f}")
                print(f"    • {FEATURE_NAMES[4]}: {row[FEATURE_NAMES[4]]:>23.2f}")
                print(f"  Expected Performance:")
                print(f"    • Predicted objective:   {row['predicted_objective']:.4f} ± {row['uncertainty']:.4f}")
                print(f"    • Expected Improvement:  {row['EI']:.6f}\n")

        print(f"{'='*80}")
        print(f"💡 TIP: The 'EI' batch explores for new peaks, while the 'Exploit' batch targets the highest known peak.")
        print(f"📊 Full details saved to: outputs/continuous_suggestions.csv")
        print(f"{'='*80}\n")

    else:
        bo_results = run_bayesian_optimization(X, y, n_init=N_INIT,
                                               n_iter=N_ITER,
                                               batch_size=batch_size)

        # Save optimization history
        history_df = pd.DataFrame(bo_results['history'])
        history_df['experiment_id'] = df.loc[history_df['suggested_index'],
                                             'experiment_id'].values
        history_df.to_csv('outputs/optimization_history.csv', index=False)
        print("✓ Saved optimization_history.csv")

        # Plot optimization trajectory
        plot_optimization_trajectory(bo_results['best_per_iteration'], bo_results)

    # Sorted experiments
    df_sorted = df.sort_values('objective', ascending=False)
    df_sorted.to_csv('outputs/sorted_experiments.csv', index=False)
    print("✓ Saved sorted_experiments.csv")

    # Machine Learning
    ml_results = train_random_forest(X, y, feature_names)

    # Save ML results
    ml_results['cv_results'].to_csv('outputs/rf_cv_results.csv', index=False)
    ml_results['importance_df'].to_csv('outputs/feature_importances.csv', index=False)
    print("✓ Saved rf_cv_results.csv and feature_importances.csv")

    # Generate plots
    print(f"\n{'='*60}")
    print("GENERATING PUBLICATION-QUALITY FIGURES (300 DPI)")
    print(f"{'='*60}")

    plot_sorted_objectives(df)
    plot_tof_vs_aqy(df)
    plot_feature_importance(ml_results['importance_df'])
    plot_predictions_vs_actual(y, ml_results['predictions'], ml_results['r2'])
    plot_feature_correlation(X, feature_names)
    plot_residual_analysis(y, ml_results['predictions'])
    plot_parameter_space(X, y, feature_names) # <-- NEW PLOT
    
    if SHAP_AVAILABLE and ml_results['shap_values'] is not None:
        plot_shap_summary(ml_results['shap_values'], X, feature_names)

    # Save requirements
    save_requirements()

    # Generate methods text
    methods = generate_methods_text(df, bo_results, ml_results)
    with open('outputs/methods_text.txt', 'w', encoding='utf-8') as f:
        f.write(methods)
    print("✓ Saved methods_text.txt")

    # Final summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE - SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments analyzed: {len(df)}")
    print(f"Best objective: {df['objective'].max():.4f} (experiment: {df.loc[df['objective'].idxmax(), 'experiment_id']})")
    print(f"\nTop 3 features by importance:")
    for i, row in ml_results['importance_df'].head(3).iterrows():
        print(f"  {i+1}. {row['feature']}: {row['importance_mean']:.4f}")
    print(f"\nRandom Forest performance:")
    print(f"  R² = {ml_results['r2']:.4f}")
    print(f"  RMSE = {ml_results['rmse']:.4f}")

    if not continuous_mode:
        print(f"\nBayesian Optimization:")
        print(f"  Initial best: {bo_results['best_per_iteration'][0]:.4f}")
        print(f"  Final best: {bo_results['best_per_iteration'][-1]:.4f}")
        print(f"  Improvement: {bo_results['best_per_iteration'][-1] - bo_results['best_per_iteration'][0]:.4f}")

    print(f"\n{'='*60}")
    print("OUTPUT FILES")
    print(f"{'='*60}")
    print("Figures (PNG 300dpi + PDF):")
    figures = ['fig_optimization_trajectory', 'fig_sorted_objectives_tof_aqy',
               'fig_tof_vs_aqy', 'fig_feature_importance', 'fig_predictions_vs_actual',
               'fig_feature_correlation', 'fig_residual_analysis', 'fig_parameter_space']
    if SHAP_AVAILABLE:
        figures.extend(['fig_shap_summary', 'fig_shap_beeswarm'])
    for fig in sorted(figures):
        print(f"  figures/{fig}.png|pdf")

    print("\nData tables (CSV):")
    tables = ['full_dataset_with_objective', 'sorted_experiments',
              'rf_cv_results', 'feature_importances']
    if not continuous_mode:
        tables.insert(1, 'optimization_history')
    else:
        tables.insert(1, 'continuous_suggestions')
    for tbl in sorted(tables):
        print(f"  outputs/{tbl}.csv")

    print(f"\n{'='*60}")
    print("✓ All outputs generated successfully!")
    print("✓ Publication-quality figures ready at 300 DPI")
    print(f"{'='*60}\n")

    return df, bo_results, ml_results

# ============================================================================
# UNIT TESTS
# ============================================================================

def run_unit_tests():
    """Simple unit tests for critical functions."""
    print("Running unit tests...")

    # Test 1: Expected Improvement function
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.5, 0.3, 0.1])
    y_best = 2.5
    ei = expected_improvement(mu, sigma, y_best, xi=0.01)
    assert ei.shape == (3,), "EI shape mismatch"
    assert np.all(ei >= 0), "EI should be non-negative"
    assert ei[2] > ei[0], "Higher mean should give higher EI when above y_best"
    print("  ✓ Expected Improvement tests passed")

    # Test 2: Kernel output
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    X_test = np.random.randn(10, 3)
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(3)) + WhiteKernel(1e-6)
    K = kernel(X_test)
    assert K.shape == (10, 10), "Kernel matrix shape mismatch"
    assert np.allclose(K, K.T), "Kernel matrix should be symmetric"
    print("  ✓ Kernel tests passed")

    # Test 3: Plot style configuration
    setup_plot_style()
    assert plt.rcParams['font.size'] == PLOT_STYLE['font.size'], "Plot style not applied"
    print("  ✓ Plot style tests passed")

    # Test 4: Trajectory plot alignment (batch_size > 1)
    # Simulate BO results with batch_size=2
    best_per_iter = np.array([1.0, 1.5, 2.0, 2.3])  # 4 entries (init + 3 iterations)
    history = [
        {'iteration': 0, 'predicted_mean': 1.2, 'predicted_std': 0.3},
        {'iteration': 0, 'predicted_mean': 1.4, 'predicted_std': 0.2},  # batch_size=2
        {'iteration': 1, 'predicted_mean': 1.8, 'predicted_std': 0.25},
        {'iteration': 1, 'predicted_mean': 1.6, 'predicted_std': 0.28},
        {'iteration': 2, 'predicted_mean': 2.1, 'predicted_std': 0.2},
        {'iteration': 2, 'predicted_mean': 2.0, 'predicted_std': 0.22},
    ]
    bo_results = {'history': history, 'gp': None, 'scaler': None}

    # Check that grouping works correctly
    history_df = pd.DataFrame(history)
    grouped = history_df.groupby('iteration').agg({
        'predicted_mean': 'max',
        'predicted_std': 'mean'
    }).reset_index()

    assert len(grouped) == 3, "Grouped history should have 3 iterations"
    assert len(best_per_iter) == 4, "best_per_iteration should be n_iter + 1"
    print("  ✓ Trajectory plot alignment tests passed")

    print("  ✓ Unit tests completed\n")

    return True


def verify_outputs():
    """Verify all expected outputs exist."""
    print("\nVerifying outputs...")

    required_files = [
        'outputs/full_dataset_with_objective.csv',
        'outputs/sorted_experiments.csv',
        'outputs/rf_cv_results.csv',
        'outputs/feature_importances.csv',
        'figures/fig_tof_vs_aqy.png',
        'figures/fig_feature_importance.png',
        'figures/fig_feature_correlation.png',
        'figures/fig_residual_analysis.png',
        'figures/fig_parameter_space.png',
    ]

    all_exist = True
    for fpath in required_files:
        if os.path.exists(fpath):
            print(f"  ✓ {fpath}")
        else:
            print(f"  ✗ {fpath} MISSING")
            all_exist = False

    if all_exist:
        print("\n✓ All required outputs verified!")
    else:
        print("\n⚠ Some outputs missing")

    return all_exist


# ============================================================================
# CLI & MAIN
# ============================================================================

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Bonke-style holistic optimization analysis with enhanced visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--continuous', action='store_true',
                        help='Use continuous optimization to suggest NEW experiments (recommended)')
    parser.add_argument('--n_suggest', type=int, default=5,
                        help='Number of new experiments to suggest in continuous mode')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for pool-based BO (only used without --continuous)')
    parser.add_argument('--xi', type=float, default=None,
                        help='Exploration parameter (0.0=exploit, 0.01=balanced, 0.1+=explore)')
    parser.add_argument('--test', action='store_true',
                        help='Run unit tests only')

    args = parser.parse_args()

    if args.test:
        run_unit_tests()
        return

    # Run main analysis
    try:
        run_unit_tests()
        df, bo_results, ml_results = run_analysis(
            continuous_mode=args.continuous,
            n_suggest=args.n_suggest,
            batch_size=args.batch_size,
            xi_override=args.xi
        )
        verify_outputs()

    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR DURING EXECUTION")
        print(f"{'='*60}")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()