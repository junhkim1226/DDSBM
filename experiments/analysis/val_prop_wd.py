import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from rdkit import Chem
from rdkit.Chem import Crippen
from scipy.stats import wasserstein_distance

from ddsbm import DATA_PATH, RESULTS_PATH


def get_LOGP(mol: Chem.Mol) -> float:
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    return Crippen.MolLogP(mol)


def epoch_to_outer_iter(epoch: int) -> int:
    return (epoch + 1) // 300 - 1


def outer_iter_to_epoch(iteration: int, n_epochs: int = 300) -> int:
    return (iteration + 1) * n_epochs - 1


def get_gen_results(path: Path, seeds: list[int], direction: str):
    dic = defaultdict(list)

    for seed in seeds:
        # NOTE: made from `graph_to_mol.py
        try:
            file = path / f"result_{direction}_seed{seed}.csv"
            gen_df = pd.read_csv(file)
        except FileNotFoundError:
            file = path / f"result_seed{seed}.csv"
            gen_df = pd.read_csv(file)

        gen_smis = gen_df["smiles"].values
        gen_success = gen_df["success"].values
        gen_idx = gen_df["idx"].values
        sorted_idx = np.argsort(gen_idx)

        dic["smiles"].extend(gen_smis[sorted_idx].tolist())
        dic["success"].extend(gen_success[sorted_idx].tolist())
        dic["idx"].extend(gen_idx[sorted_idx].tolist())

    for key, value in dic.items():
        dic[key] = np.array(value)

    return dic


def get_prop(mol_or_smi: Union[str, Chem.Mol], prop: str) -> float:
    mol = mol_or_smi
    if isinstance(mol_or_smi, str):
        mol = Chem.MolFromSmiles(mol_or_smi)

    if prop == "qed":
        return Chem.QED.qed(mol)
    elif prop == "logp":
        try:
            return Crippen.MolLogP(mol)
        except:
            return None
    else:
        raise ValueError(f"Invalid property: {prop}")


def get_mol(smi: Optional[str]) -> Chem.Mol:
    """
    Loads SMILES/molecule into RDKit's object
    """
    if smi is None:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return mol


def main():
    experiment_path = args.experiment_path.resolve()
    exp_name = experiment_path.name
    dataset_name = experiment_path.parent.name

    seeds = sorted(set(args.seeds))
    iterations = sorted(set(args.iterations))

    data_dir = DATA_PATH / dataset_name / exp_name
    original_train_data_path = data_dir / "raw" / "train.csv"
    original_test_data_path = data_dir / "raw" / "test.csv"
    assert original_train_data_path.exists(), f"{original_train_data_path} not found"
    assert original_test_data_path.exists(), f"{original_test_data_path} not found"
    original_train_df = pd.read_csv(original_train_data_path)
    original_test_df = pd.read_csv(original_test_data_path)
    original_test_idx = original_test_df["Unnamed: 0"].values
    original_test_idx = np.concatenate(
        [original_test_idx for _ in range(len(seeds))], axis=0
    )

    if args.bridge:
        # NOTE: We compare bridge and SB by aligning their results in terms of total epochs
        iterations = [
            outer_iter_to_epoch(iteration, args.n_epochs) for iteration in iterations
        ]
    experiment_paths = [
        experiment_path / f"test_{args.direction}_{iteration}_last"
        for iteration in iterations
    ]

    assert all(experiment_path.exists() for experiment_path in experiment_paths), (
        f"some paths in {experiment_paths} are not found"
    )

    if args.direction == "forward":
        target = "XT"
        source = "X0"
        target_key = "PRB"
        source_key = "REF"
    else:
        target = "X0"
        source = "XT"
        target_key = "REF"
        source_key = "PRB"

    wd_dic = defaultdict(list)
    val_dic = defaultdict(list)
    with parallel_backend("loky", n_jobs=args.num_workers):
        # NOTE: Data duplication does not affect WD
        original_target_props = Parallel()(
            delayed(get_prop)(smi, "logp")
            for smi in original_test_df[f"{target_key}-SMI"].values
        )
        original_target_props = np.array(original_target_props * len(seeds))
        original_source_props = Parallel()(
            delayed(get_prop)(smi, "logp")
            for smi in original_test_df[f"{source_key}-SMI"].values
        )
        original_source_props = np.array(original_source_props * len(seeds))

        original_props_wd = wasserstein_distance(
            original_target_props, original_source_props
        )

        for experiment_path, iteration in zip(experiment_paths, iterations):
            gen_results = get_gen_results(experiment_path, seeds, args.direction)
            assert len(original_test_idx) == len(gen_results["idx"])
            assert np.all(original_test_idx == gen_results["idx"]), "IDX MISMATCH"

            gen_props = Parallel()(
                delayed(get_prop)(smi, "logp")
                for smi in gen_results["smiles"][gen_results["success"]]
            )
            gen_props = np.array(gen_props)
            gen_wd = wasserstein_distance(original_target_props, gen_props)

            wd_dic["iteration"].append(iteration)
            wd_dic["sb"].append(gen_wd)
            wd_dic["test_x0_test_xt"].append(original_props_wd)

            val_dic["iteration"].append(iteration)
            val_dic["sb"].append(gen_results["success"].mean())

    savedir = args.output_path.resolve()
    savedir.mkdir(exist_ok=True, parents=True)
    seed_info = "_".join(list(map(str, seeds)))

    wd_df = pd.DataFrame(wd_dic)
    output_file = savedir / f"{dataset_name}-{exp_name}-{args.direction}-wd-{seed_info}.csv"
    wd_df.to_csv(output_file, index=False)
    print("SAVED WD RESULT IN", output_file)

    val_df = pd.DataFrame(val_dic)
    output_file = savedir / f"{dataset_name}-{exp_name}-{args.direction}-val-{seed_info}.csv"
    val_df.to_csv(output_file, index=False)
    print("SAVED VAL RESULT IN", output_file)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_path",
        type=Path,
        help="Experiment directory, which consists of various directories "
        "that containing result csvs generated by `analysis/graph_to_mol.py`",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Directory in which output files will be written to",
        default=RESULTS_PATH / "val_props",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[0, 4, 9],
        help="List of iterations to be evaluated. Each iteration corresponds to an epoch."
        "Default: [0, 4, 9]",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=[42],
        nargs="+",
        help="list of seed numbers (default: 42)",
    )
    parser.add_argument(
        "--bridge",
        action="store_true",
        help="If true, use bridge experiment. Default: false",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=300,  # NOTE: zinc - 300 / polymer - 250
        help="Number of epochs of respective SB experiment. Only needed when --bridge is true",
    )
    parser.add_argument(
        "--property",
        type=str,
        default="logp",
        help="Property used to compute Wasserstein Distance",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="forward",
        choices=["forward", "backward"],
        help="Direction of the experiment. Default: forward",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for process. Default: 4",
    )
    args = parser.parse_args()

    main()
