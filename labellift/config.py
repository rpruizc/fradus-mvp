"""Global configuration constants for the LabelLift MVP.

Every script in this package is deterministic and seeded from ``RANDOM_SEED`` so
that all outputs are fully reproducible.
"""

from pathlib import Path

# Reproducibility ----------------------------------------------------------------
RANDOM_SEED = 42

# Synthetic dataset --------------------------------------------------------------
N_TRANSACTIONS = 1_000_000
TRAINING_WINDOW_DAYS = 30

# Label corruption (miscoding) rates ---------------------------------------------
LABEL_CORRUPTION_FALSE_NEGATIVE = 0.06
LABEL_CORRUPTION_FALSE_POSITIVE = 0.08

# Propensity clipping bounds -----------------------------------------------------
PROPENSITY_FLOOR = 0.05
PROPENSITY_CEILING = 0.98

# Filesystem paths ---------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

SYNTHETIC_TRANSACTIONS_CSV = DATA_DIR / "synthetic_transactions.csv"
CORRECTED_LABELS_CSV = DATA_DIR / "corrected_labels.csv"
BACKTEST_METRICS_CSV = DATA_DIR / "backtest_metrics.csv"
