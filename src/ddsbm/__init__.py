from pathlib import Path

from ddsbm import graph_match_helper, train_helper

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "configs"
DATA_PATH: Path = PROJECT_ROOT / "data"
RESULTS_PATH: Path = PROJECT_ROOT / "results"
OUTPTUS_PATH: Path = PROJECT_ROOT / "outputs"
EXECUTABLES: dict[str, str] = {
    "data_generation": train_helper.__file__,
    "train": train_helper.__file__,
    "test": train_helper.__file__,  # NOTE: test is same as train
    "graph_match": graph_match_helper.__file__,
}
