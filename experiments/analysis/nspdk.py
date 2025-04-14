"""
Make conda environment:
>>> conda create -c conda-forge -n NSPDK rdkit=2023.03.2 python=3.9
>>> pip install git+https://github.com/fabriziocosta/EDeN.git --user
>>> pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
"""

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from eden.graph import Vectorizer, vectorize
from eden.ml.ml import multiprocess_vectorize
from joblib import parallel_backend
from rdkit import Chem
from sklearn.metrics.pairwise import pairwise_kernels

try:
    from ddsbm import DATA_PATH, RESULTS_PATH
except ImportError:
    DATA_PATH = Path(__file__).parents[2] / "data"
    RESULTS_PATH = Path(__file__).parents[2] / "results"

# Check for fixed PYTHONHASHSEED (randomness in eden.graph.vectorize)
PYTHONHASHSEED = os.environ.get("PYTHONHASHSEED")
if not PYTHONHASHSEED:
    print("Warning: PYTHONHASHSEED is not fixed. Set it for reproducibility.")
else:
    print(f"PYTHONHASHSEED: {PYTHONHASHSEED}")

# Configure numpy output options
np.set_printoptions(threshold=np.inf)

# Dictionary for bond types
BOND_TYPE_DICT = {
    None: 0,
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}


# From https://github.com/yunhuijang/HGGT/blob/a9d695666f16e2cf39adf4efbf175ee0df75b55d/data/mol_utils.py
def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), label=atom.GetSymbol())
            # G.add_node(atom.GetIdx(),
            #            label=NODE_TYPE_DICT[atom.GetSymbol()])
        for bond in mol.GetBonds():
            # G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label=int(bond.GetBondTypeAsDouble()))
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                label=BOND_TYPE_DICT[bond.GetBondType()],
            )
            # label=BOND_TYPE_DICT[bond.GetBondTypeAsDouble()])
        nx_graphs.append(G)
    return nx_graphs


def smis_to_nx(smis):
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    nx_graphs = mols_to_nx(mols)
    return nx_graphs


def timer(func):
    def wrapper(*args, **kwargs):
        # Measure the length of inputs
        input_lengths = {
            f"arg_{i}": len(arg) if hasattr(arg, "__len__") else "N/A"
            for i, arg in enumerate(args)
        }

        # Start timing
        start_time = time.time()

        # Execute the function
        result = func(*args, **kwargs)

        # End timing
        execution_time = time.time() - start_time

        # Print input lengths and execution time
        print(f"Input lengths: {input_lengths}")
        print(f"Execution time: {execution_time:.6f} seconds")

        return result

    return wrapper


### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/mmd.py
@timer
def compute_nspdk_mmd(samples1, samples2, metric="nspdk", is_hist=False, n_jobs=20):
    def kernel_compute(X, Y=None, is_hist=True, metric="linear", n_jobs=None):
        X = vectorize(X, complexity=4, discrete=True)
        if Y is not None:
            Y = vectorize(Y, complexity=4, discrete=True)
        return pairwise_kernels(X, Y, metric="linear", n_jobs=n_jobs)

    def multiprocess_kernel_compute(
        X, Y=None, is_hist=True, metric="linear", n_jobs=None
    ):
        vectorizer = Vectorizer(complexity=4, discrete=True)
        X = multiprocess_vectorize(X, vectorizer, n_jobs=n_jobs, n_blocks=n_jobs - 1)
        if Y is not None:
            Y = multiprocess_vectorize(
                Y, vectorizer, n_jobs=n_jobs, n_blocks=n_jobs - 1
            )
        return pairwise_kernels(X, Y, metric="linear", n_jobs=n_jobs)

    worker = multiprocess_kernel_compute if n_jobs > 1 else kernel_compute
    X = worker(samples1, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Y = worker(samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Z = worker(samples1, Y=samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)

    return np.average(X) + np.average(Y) - 2 * np.average(Z)


def get_gen_results(path: Path, seeds: list[int], direction: str):
    dic = defaultdict(list)

    for seed in seeds:
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
    return (iteration + 1) * n_epochs - 1  # NOTE: ZINC - 300, POLYMER - 250


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

    # Converting SMILES to NetworkX graphs for NSPDK calculation
    original_train_target_smiles = original_train_df[f"{target_key}-SMI"].values
    original_test_target_smiles = original_test_df[f"{target_key}-SMI"].values
    original_train_target_graphs = smis_to_nx(original_train_target_smiles)
    original_test_target_graphs = smis_to_nx(original_test_target_smiles)

    original_train_source_smiles = original_train_df[f"{source_key}-SMI"].values
    original_test_source_smiles = original_test_df[f"{source_key}-SMI"].values
    original_train_source_graphs = smis_to_nx(original_train_source_smiles)
    original_test_source_graphs = smis_to_nx(original_test_source_smiles)

    print("INITIAL DATA PREPARATION DONE")
    test_xt_train_xt = compute_nspdk_mmd(
        original_test_target_graphs,
        original_train_target_graphs,
        n_jobs=args.num_workers,
    )

    nspdk_dic = defaultdict(list)
    with parallel_backend("loky", n_jobs=args.num_workers):
        for experiment_path, iteration in zip(experiment_paths, iterations):
            gen_results = get_gen_results(experiment_path, seeds, args.direction)
            gen_mols = [
                Chem.MolFromSmiles(smi)
                for smi in gen_results["smiles"][gen_results["success"]]
                if smi
            ]

            gen_graphs = smis_to_nx(gen_results["smiles"][gen_results["success"]])
            nspdk = compute_nspdk_mmd(
                original_train_target_graphs, gen_graphs, n_jobs=args.num_workers
            )
            nspdk_dic["iteration"].append(iteration)
            nspdk_dic["sb"].append(nspdk)
            nspdk_dic["ref"].append(test_xt_train_xt)

    savedir = args.output_path.resolve()
    savedir.mkdir(exist_ok=True, parents=True)
    fcd_df = pd.DataFrame(nspdk_dic)
    seed_info = "_".join(list(map(str, seeds)))
    output_file = savedir / f"{dataset_name}-{exp_name}-{args.direction}-{seed_info}.csv"
    fcd_df.to_csv(output_file, index=False)
    print("SAVED NSPDK RESULT IN", output_file)
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
        default=RESULTS_PATH / "nspdk",
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
        help="Direction of the experiment. Default: forward",
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
        help="Number of workers for NSPDK. Default: 1",
    )
    args = parser.parse_args()

    main()
