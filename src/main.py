import datetime
import logging
import os
import subprocess
import sys
import tempfile
from enum import Enum
from itertools import product
from pathlib import Path
from shutil import copy2, copytree

import hydra
from omegaconf import DictConfig, OmegaConf

from ddsbm import CONFIG_PATH, EXECUTABLES, PROJECT_ROOT


class DIRECTIONS(Enum):
    B = "backward"
    F = "forward"

    def __init__(self, value: str):
        if value not in ["backward", "forward"]:
            raise ValueError("Invalid direction value")
        self._value_ = value

    @property
    def initial(self) -> str:
        return self._value_

    @property
    def last(self) -> str:
        direc = DIRECTIONS.F if self == DIRECTIONS.B else DIRECTIONS.B
        return direc.value

    def opposite(self, direc: str) -> str:
        return self.F.value if direc == self.B.value else self.B.value


def get_logger(filename: str, level=logging.INFO):
    """Get the logger
    Args:
        filename (str): filename of the logger
        level (int): logging level
    Return:
        logger (logging.Logger): logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(f"{filename}.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _copy_data(src: Path, dst: Path):
    copytree(src / "raw", dst / "raw", dirs_exist_ok=True)
    for file in src.glob("*.txt"):
        copy2(file, dst)
    return


def run_data_generation(cfg: DictConfig, logger: logging.Logger):
    _cfg = cfg.copy()
    cfg_data = _cfg.dataset

    datadir = PROJECT_ROOT / cfg_data.datadir
    new_datadir = datadir / cfg_data.datadirname
    original_datadir = datadir / cfg_data.original_datadirname
    assert original_datadir.exists(), (
        f"Original datadir {original_datadir} does not exist"
    )
    logger.info(f"Copy the data from {original_datadir} to {new_datadir}")
    _copy_data(original_datadir, new_datadir)

    logger.info("Generate Training Data")
    cfg_data.datadir = str(new_datadir)
    cfg_data.compute_dataset_infos = True
    run_subprocess(_cfg, logger, "data_generation")

    cfg_data.compute_dataset_infos = False

    if cfg_data.unconditional:
        # NOTE : Unconditional Prior Sample (junhkim)
        print(f"Debug] {cfg_data.prior_sample}")
        run_subprocess(_cfg, logger, "data_generation")

    return str(new_datadir)


def set_experiment_cfg(
    cfg: DictConfig, iteration: int, direction: str, whole_exp_wd: Path
):
    _cfg = cfg.copy()

    _cfg.experiment.current_direction = direction
    _cfg.experiment.current_iteration = iteration
    _cfg.train.bridge_direction = direction
    # NOTE: Generation data path for graph matching
    _cfg.graph_match.data_path = str(whole_exp_wd / f"{direction}_{iteration}")
    _cfg.graph_match.iteration = iteration

    # NOTE:should load resume path
    if iteration > 0:
        prev_exp_path = whole_exp_wd / f"{direction}_{iteration - 1}"
        resume_path = prev_exp_path / "checkpoints/last.ckpt"
        _cfg.general.resume = str(resume_path)
    return _cfg


def run_graph_match(cfg: DictConfig, logger: logging.Logger, test: bool = False):
    """Run graph matching

    1. Initial graph matching - no iteration and direction
    2. Training graph matching - with iteration and direction, end of training
    3. Test graph matching - with iteration and direction, end of training
    """
    _cfg = cfg.copy()
    _cfg.graph_match.gpus = _cfg.general.gpus

    # NOTE: Test
    if test:
        _cfg.graph_match.test = True
        # NOTE: Graph matching internally use same datamodule with training
        # -> should set sample_batch_size in test case
        _cfg.general.sample_batch_size = _cfg.graph_match.batch_size
        # TODO: Check if explicitly setting config.model.* is handled
        # This seems necessary since we should graph match with
        # the same condition to measure NLL

        _cfg.graph_match.pooling_type = "max"
        _cfg.graph_match.dtype = "float32"
        _cfg.graph_match.noise_coeff = 1e-6
        _cfg.graph_match.max_iter = 2500
        _cfg.graph_match.tol = 1e-4
        _cfg.graph_match.num_seed = 10
    else:
        _cfg.graph_match.test = False
        # NOTE: Graph matching internally use same datamodule with training
        # -> should set sample_batch_size in train case
        _cfg.train.batch_size = _cfg.graph_match.batch_size

    run_subprocess(_cfg, logger, "graph_match")
    return


def run_single_bridge(cfg: DictConfig, logger: logging.Logger, iteration: int):
    # NOTE: Should handle locally due to use same iteration in both directions
    _cfg = cfg.copy()
    _cfg.train.n_epochs = _cfg.train.n_epochs * (iteration + 1)

    run_subprocess(_cfg, logger, "train")
    return


def run_subprocess(cfg: DictConfig, logger: logging.Logger, run_type: str):
    assert run_type in EXECUTABLES, f"Invalid run type {run_type}"
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"ddsbm_{run_type}_",
        suffix=".yaml",
        dir="/tmp",
        delete=False,
    ) as tmp:
        OmegaConf.save(cfg, tmp.name, resolve=True)
        config_path = tmp.name

    iteration = cfg.experiment.current_iteration
    direction = cfg.experiment.current_direction
    if iteration is not None:
        msg = f"Running {run_type} {iteration} in {direction} direction"
    else:
        msg = f"Running {run_type}"
    logger.info(msg)

    cmd = [sys.executable, EXECUTABLES[run_type], config_path]
    try:
        result = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            if "statistics done" in result.stderr:
                logger.info("Data generation: 'statistics done' exception ignored.")
            elif "sample done" in result.stderr:
                logger.info("Data generation: 'sample done' exception ignored.")
            else:
                raise AssertionError(f"Subprocess failed: {result.stderr}")
    except Exception as e:
        raise e
    finally:
        os.unlink(config_path)

    logger.info(f"{msg} is done")
    return


@hydra.main(
    version_base="1.3", config_path=str(CONFIG_PATH), config_name="config_train"
)
def train(cfg: DictConfig):
    cfg_exp = cfg.experiment
    ISBRIDGE = cfg_exp.outer_loops == 1 and cfg_exp.single_direction

    if cfg_exp.outer_loops == 1:
        assert cfg_exp.initial_direction == cfg_exp.current_direction, (
            "Initial direction should be the same as current direction for bridge"
        )

    directions = DIRECTIONS(cfg_exp.initial_direction.lower())
    initial_direction = directions.initial
    last_direction = directions.last

    logger = get_logger(filename=str(Path(cfg.general.name).resolve()))
    all_experiments = list(
        product(
            range(cfg_exp.outer_loops),
            [initial_direction, last_direction],
        )
    )
    try:
        previously_done_exp_index = all_experiments.index(
            (cfg_exp.current_iteration, cfg_exp.current_direction)
        )
    except ValueError:
        # NOTE: STARTING FROM THE BEGINNING
        previously_done_exp_index = -1

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    if cfg.general.prepend_date_in_name:
        cfg.general.name = f"{date}_{cfg.general.name}"

    # /home/mseok/work/DL/DDSBM/REFACTORING
    if not cfg_exp.skip_data_generation:
        # Run data generation
        new_datadir = run_data_generation(cfg, logger)
        if cfg.dataset.unconditional:
            # NOTE : Finish Unconditional Prior Sample (junhkim)
            cfg.dataset.prior_sample = False
    else:
        datadir = PROJECT_ROOT / cfg.dataset.datadir
        new_datadir = str(datadir / cfg.dataset.datadirname)
    cfg.dataset.datadir = new_datadir

    whole_exp_wd = PROJECT_ROOT / "outputs" / cfg.dataset.name / cfg.general.name
    OmegaConf.save(cfg, whole_exp_wd / ".hydra" / "updated_config.yaml", resolve=True)

    if not cfg_exp.skip_initial_graph_matching:
        # Run initial graph matching
        run_graph_match(cfg, logger, False)  # TRAIN
        run_graph_match(cfg, logger, True)  # TEST

    for exp_idx, (iteration, direction) in enumerate(all_experiments):
        # NOTE: Assuming that we always start from already done experiment
        # Thus we skip all previous experiments based on subsequent experiemtn of cfg
        if exp_idx < previously_done_exp_index + 1:
            continue
        _cfg = set_experiment_cfg(cfg, iteration, direction, whole_exp_wd)
        sub_exp_wd = whole_exp_wd / f"{direction}_{iteration}"
        sub_exp_wd.mkdir(exist_ok=True, parents=True)

        # /home/mseok/work/DL/DDSBM/REFACTORING/outputs/zinc/SB_R_0.999_U/backward_0
        os.chdir(sub_exp_wd)

        # Train
        run_single_bridge(_cfg, logger, iteration)
        (sub_exp_wd / ".hydra").mkdir(exist_ok=True)
        OmegaConf.save(_cfg, sub_exp_wd / ".hydra/config.yaml", resolve=True)

        # Graph match
        if not cfg_exp.skip_graph_matching:
            run_graph_match(_cfg, logger)

        os.chdir(whole_exp_wd)

    exp = "Bridge" if ISBRIDGE else "SB"
    logger.info(f"{exp} Training is done")
    return


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="config_test")
def test(cfg: DictConfig):
    _cfg = cfg.copy()
    ckpt_path = Path(_cfg.general.test_only).resolve()
    _cfg.general.test_only = ckpt_path
    assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist"
    # outputs/zinc/2025-04-06_SB_R_0.999_uniform_42/backward_0/checkpoints/last.ckpt
    train_cfg_path = ckpt_path.parents[1] / ".hydra" / "config.yaml"
    assert train_cfg_path.exists(), f"Train config path {train_cfg_path} does not exist"
    cfg_train = OmegaConf.load(train_cfg_path)
    _cfg.dataset = OmegaConf.merge(_cfg.dataset, cfg_train.dataset)
    iteration = cfg_train.experiment.current_iteration
    direction = cfg_train.experiment.current_direction
    _cfg.general.name = f"test_{direction}_{iteration}_{ckpt_path.stem}"

    _cfg.train.bridge_direction = direction
    _cfg.graph_match.iteration = iteration

    gen_data_path = ckpt_path.parents[2] / _cfg.general.name
    gen_data_path.mkdir(exist_ok=True, parents=True)
    _cfg.graph_match.data_path = gen_data_path
    os.chdir(_cfg.graph_match.data_path)

    logger = get_logger(filename=str(Path(_cfg.general.name).resolve()))
    run_subprocess(_cfg, logger, "test")
    run_graph_match(_cfg, logger, test=True)
    return


if __name__ == "__main__":
    train()
