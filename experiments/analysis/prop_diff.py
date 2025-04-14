import argparse
import os
import sys
from collections import defaultdict
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend

# from molskill.scorer import MolSkillScorer
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors

sys.path.append(os.path.join(os.environ["CONDA_PREFIX"], "share", "RDKit", "Contrib"))
from SA_Score import sascorer

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


def getMolDescriptors(mol, missingVal=None):
    """calculate the full list of descriptors for a molecule

    missingVal is used if the descriptor cannot be calculated
    """
    res = {}
    for nm, fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback

            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res


def get_qed(mol) -> float:
    try:
        qed = QED.qed(mol)
        return qed
    except:
        return None


def get_sa(mol) -> float:
    try:
        sa = sascorer.calculateScore(mol)
        return sa
    except:
        return None


def analyze_data_diff(inp):
    try:
        (
            iteration,
            sb_gen_smi,
            sb_gen_success,
            bridge_gen_smi,
            bridge_gen_success,
            original_source_smi,
            original_target_smi,
            idx,
        ) = inp
        only_sb = False
    except ValueError:
        (
            iteration,
            sb_gen_smi,
            sb_gen_success,
            original_source_smi,
            original_target_smi,
            idx,
        ) = inp
        only_sb = True

    original_source_mol = Chem.MolFromSmiles(original_source_smi)
    original_source_res = dict(
        # logp=get_LOGP(original_source_mol),
        QED=get_qed(original_source_mol),
        SA=get_sa(original_source_mol),
    )

    original_target_mol = Chem.MolFromSmiles(original_target_smi)
    original_target_res = dict(
        # logp=get_LOGP(original_target_mol),
        QED=get_qed(original_target_mol),
        SA=get_sa(original_target_mol),
    )

    dic = defaultdict(list)

    for key in original_source_res:
        original_source_val = original_source_res[key]
        original_target_val = original_target_res[key]
        dic["prop_name"].append(key)
        dic["prop_diff"].append(original_target_val - original_source_val)
        dic["data_type"].append("Original")
        dic["iteration"].append(iteration)

    if sb_gen_success:
        sb_gen_mol = Chem.MolFromSmiles(sb_gen_smi)
        sb_gen_res = dict(
            # logp=get_LOGP(original_target_mol),
            QED=get_qed(sb_gen_mol),
            SA=get_sa(sb_gen_mol),
        )
        for key in sb_gen_res:
            original_source_val = original_source_res[key]
            dic["prop_name"].append(key)
            if sb_gen_res[key] is None:
                dic["prop_diff"].append(None)
            else:
                dic["prop_diff"].append(sb_gen_res[key] - original_source_val)
            dic["data_type"].append("SB")
            dic["iteration"].append(iteration)
    else:
        for key in original_source_res:
            dic["prop_name"].append(key)
            dic["prop_diff"].append(None)
            dic["data_type"].append("SB")
            dic["iteration"].append(iteration)

    if not only_sb:
        if bridge_gen_success:
            bridge_gen_mol = Chem.MolFromSmiles(bridge_gen_smi)
            bridge_gen_res = dict(
                # logp=get_LOGP(original_target_mol),
                QED=get_qed(bridge_gen_mol),
                SA=get_sa(bridge_gen_mol),
            )
            for key in bridge_gen_res:
                original_source_val = original_source_res[key]
                dic["prop_name"].append(key)
                if bridge_gen_res[key] is None:
                    dic["prop_diff"].append(None)
                else:
                    dic["prop_diff"].append(bridge_gen_res[key] - original_source_val)
                dic["data_type"].append("Bridge")
                dic["iteration"].append(iteration)
        else:
            for key in original_source_res:
                dic["prop_name"].append(key)
                dic["prop_diff"].append(None)
                dic["data_type"].append("Bridge")
                dic["iteration"].append(iteration)

    dic["idx"] = [idx] * len(dic["prop_name"])
    return dic


def smi2mol(smi):
    try:
        return Chem.MolFromSmiles(smi)
    except:
        return None


def main():
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

    prop_dic = defaultdict(list)
    data_dic = defaultdict(list)  # FINAL DF
    ratio_dic = defaultdict(list)  # FINAL DF
    inputs = []
    with parallel_backend("loky", n_jobs=args.num_workers):
        for experiment_path, iteration in zip(experiment_paths, iterations):
            gen_results = get_gen_results(experiment_path, seeds, args.direction)

            assert len(original_test_idx) == len(gen_results["idx"])
            assert np.all(original_test_idx == gen_results["idx"]), "IDX MISMATCH"

            original_source_smis = original_test_df[f"{source_key}-SMI"].values
            original_source_smis = np.array(original_source_smis.tolist() * len(seeds))
            original_target_smis = original_test_df[f"{target_key}-SMI"].values
            original_target_smis = np.array(original_target_smis.tolist() * len(seeds))
            idx = original_test_df["Unnamed: 0"].values
            idx = np.array(idx.tolist() * len(seeds))

            inputs.extend(
                zip(
                    repeat(iteration),
                    gen_results["smiles"],
                    gen_results["success"],
                    original_source_smis,
                    original_target_smis,
                    idx,
                )
            )
            print("ITERATION: ", iteration, "LEN TOTAL INPUTS: ", len(inputs))

        prefix = "diff_"
        prop = "prop_diff"
        results = Parallel()(delayed(analyze_data_diff)(inp) for inp in inputs)
        for result in results:
            for key in result:
                prop_dic[key].extend(result[key])

    prop_df = pd.DataFrame(prop_dic)

    savedir = args.output_path.resolve()
    savedir.mkdir(exist_ok=True, parents=True)
    prop_df.to_csv(savedir / f"raw_{prefix}prop.csv", index=False)
    for iteration in iterations:
        partial_df = prop_df[prop_df["iteration"] == iteration]
        for prop_name in partial_df["prop_name"].unique():
            prop_partial_df = partial_df[partial_df["prop_name"] == prop_name]
            for data_type in prop_partial_df["data_type"].unique():
                data_partial_df = prop_partial_df[
                    prop_partial_df["data_type"] == data_type
                ]
                vals = data_partial_df[prop].values
                total_len = len(vals)
                vals = np.abs(vals[~np.isnan(vals)])
                data_dic[f"{data_type}_{prop_name}_diff"].append(vals.mean())

    seed_info = "_".join(list(map(str, seeds)))
    output_file = (
        savedir / f"{dataset_name}-{exp_name}-{args.direction}-{seed_info}.csv"
    )
    data_df = pd.DataFrame(data_dic)
    data_df.to_csv(output_file, index=False)
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
        default=RESULTS_PATH / "prop_diff",
        help="Directory in which output files will be written to",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for parallel processing. Default: 1",
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
        "--direction",
        type=str,
        default="forward",
        choices=["forward", "backward"],
        help="Direction of the experiment. Default: forward",
    )
    args = parser.parse_args()

    main()
