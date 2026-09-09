from pathlib import Path


# ============================================================
# SMARTENERGY NEXUS - PROJECT CONFIGURATION
# ============================================================

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# Project directories
# ------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

LOG_DIR = PROJECT_ROOT / "logs"


# ------------------------------------------------------------
# SGSC dataset
# ------------------------------------------------------------

SGSC_FILE = Path(
    r"C:\Users\nutha\Downloads\zm4f727vvr-1"
    r"\interval load data\data_3_groups.csv"
)


# ------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------

# SGSC-derived dataset uses 30-minute intervals
INTERVAL_MINUTES = 30

INTERVALS_PER_HOUR = 2
INTERVALS_PER_DAY = 48
INTERVALS_PER_WEEK = 336


# ------------------------------------------------------------
# Machine learning configuration
# ------------------------------------------------------------

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ------------------------------------------------------------
# Large-file processing
# ------------------------------------------------------------

# Number of CSV rows processed at one time
CHUNK_SIZE = 50_000