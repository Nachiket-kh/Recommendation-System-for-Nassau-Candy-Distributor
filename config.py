"""Central project configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "Nassau Candy Distributor.csv"
FACTORY_PATH = ROOT / "data" / "raw" / "factories.csv"
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.2
SCORING_WEIGHTS = {"prediction": 0.55, "cost": 0.20, "profit": 0.15, "delivery": 0.10}
