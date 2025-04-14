import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fcd_torch import FCD
from joblib import Parallel, delayed, parallel_backend
from rdkit import Chem

from ddsbm import DATA_PATH, RESULTS_PATH


def reset_atommap(mol: Chem.RWMol) -> Chem.RWMol:
    """
    Resets atom map numbers to 0.
    """

    for idx, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(0)
    return mol


def reset_atommap_smi(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return
    mol = reset_atommap(mol)
    return Chem.MolToSmiles(mol)


def get_gen_results(path: Path, seeds: list[int], direction: str):
    dic = defaultdict(list)

    for seed in seeds:
        # NOTE: made from `graph_to_mol.py
        file = path / f"result_{direction}_seed{seed}.csv"
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


def outer_iter_to_epoch(iteration: int, n_epochs: int = 300) -> int:
    return (iteration + 1) * n_epochs - 1


def get_fcd_results(
    fcd_calculator: FCD,
    num_workers: int,
    seeds: list[int],
    iterations: list[int],
    p_target: dict[str, torch.Tensor],
    experiment_paths: list[Path],
):
    """
    Compute fcd results for all experiments
    """

    fcd_dic = defaultdict(list)
    with parallel_backend("loky", n_jobs=num_workers):
        for experiment_path, iteration in zip(experiment_paths, iterations):
            print("OUTER IDX: ", iteration)
            print("SB EXP PATH: ", experiment_path)

            sb_gen_results = get_gen_results(experiment_path, seeds, args.direction)
            sb_gen_smis = Parallel()(
                delayed(reset_atommap_smi)(smi)
                for smi in sb_gen_results["smiles"][sb_gen_results["success"]]
            )
            sb_gen_smis = [smi for smi in sb_gen_smis if smi is not None]

            if len(sb_gen_smis) < 10000:
                print(
                    f"Number of SB generated SMILES is less than 10,000: {len(sb_gen_smis)}"
                )
            sb_gen_fcd = fcd_calculator(pref=p_target, gen=sb_gen_smis)

            fcd_dic["iteration"].append(iteration)
            fcd_dic["sb"].append(sb_gen_fcd)
    return fcd_dic


def main():
    # NOTE: FCD calculator
    if torch.cuda.is_available():
        fcd_calculator = FCD(device="cuda", n_jobs=args.num_workers)
    else:
        fcd_calculator = FCD(device="cpu", n_jobs=args.num_workers)

    experiment_path = args.experiment_path.resolve()
    exp_name = experiment_path.name
    dataset_name = experiment_path.parent.name

    seeds = sorted(set(args.seeds))
    iterations = sorted(set(args.iterations))

    data_dir = DATA_PATH / dataset_name / exp_name
    # NOTE: We use train data to compute FCD
    original_train_data_path = data_dir / "raw" / "train.csv"
    original_test_data_path = data_dir / "raw" / "test.csv"
    assert original_train_data_path.exists(), f"{original_train_data_path} not found"
    assert original_test_data_path.exists(), f"{original_test_data_path} not found"

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

    original_train_df = pd.read_csv(original_train_data_path)
    original_test_df = pd.read_csv(original_test_data_path)

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

    original_train_target_smiles = original_train_df[f"{target_key}-SMI"].values
    original_test_target_smiles = original_test_df[f"{target_key}-SMI"].values
    p_target = fcd_calculator.precalc(original_train_target_smiles)

    original_train_source_smiles = original_train_df[f"{source_key}-SMI"].values
    original_test_source_smiles = original_test_df[f"{source_key}-SMI"].values
    p_test_source = fcd_calculator.precalc(original_test_source_smiles)

    train_x0_train_xt = fcd_calculator(pref=p_target, gen=original_train_source_smiles)
    test_x0_train_xt = fcd_calculator(pref=p_target, gen=original_test_source_smiles)
    test_x0_test_xt = fcd_calculator(
        ref=original_test_target_smiles, gen=original_test_source_smiles
    )
    test_xt_train_xt = fcd_calculator(pref=p_target, gen=original_test_target_smiles)

    fcd_dic = get_fcd_results(
        fcd_calculator=fcd_calculator,
        num_workers=args.num_workers,
        seeds=seeds,
        iterations=iterations,
        p_target=p_target,
        experiment_paths=experiment_paths,
    )
    fcd_dic["test_x0_train_xt"] = [test_x0_train_xt] * len(iterations)
    fcd_dic["test_xt_train_xt"] = [test_xt_train_xt] * len(iterations)

    savedir = args.output_path.resolve()
    savedir.mkdir(exist_ok=True, parents=True)
    fcd_df = pd.DataFrame(fcd_dic)
    seed_info = "_".join(list(map(str, seeds)))
    output_file = (
        savedir / f"{dataset_name}-{exp_name}-{args.direction}-{seed_info}.csv"
    )
    fcd_df.to_csv(output_file, index=False)
    print("SAVED FCD RESULT IN", output_file)
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
        default=RESULTS_PATH / "fcd",
        help="Directory in which output files will be written to",
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
        "--direction",
        type=str,
        default="forward",
        choices=["forward", "backward"],
        help="Direction of SB or Bridge experiments, default: forward",
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
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for FCD. Default: 1",
    )
    args = parser.parse_args()

    main()
