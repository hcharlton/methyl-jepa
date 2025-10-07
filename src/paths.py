from pathlib import Path

def get_project_root():
    current_path = Path.cwd()
    while current_path.parent != current_path:
        if (current_path / ".git").exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("Project root not found. No .git directory")

PROJECT_ROOT = get_project_root()

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_DIR = DATA_DIR / "00_raw"
PROCESSED_DATA_DIR = DATA_DIR / "01_processed"
ANALYSIS_DIR = DATA_DIR / "02_analysis"

INFERENCE_DATA_DIR = PROCESSED_DATA_DIR/'inference_sets'

