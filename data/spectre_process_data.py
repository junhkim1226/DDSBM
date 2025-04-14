import random
import sys
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from ddsbm import DATA_PATH

SEED = 42
random.seed(SEED)


def main():
    """
    For Unconditional Generation, Drop lines related to "REF-SMI"
    """
    adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = (
        torch.load(args.file)
    )
    num_graphs = len(adjs)

    g_cpu = torch.Generator()
    g_cpu.manual_seed(0)

    test_len = int(round(num_graphs * 0.2))
    train_len = int(round((num_graphs - test_len) * 0.8))
    val_len = num_graphs - train_len - test_len

    indices = torch.randperm(num_graphs, generator=g_cpu)

    # train_indices = indices[:train_len]
    # val_indices = indices[train_len-10:train_len]

    train_indices = indices[:train_len]
    val_indices = indices[train_len : train_len + val_len]
    test_indices = indices[train_len + val_len :]
    # NOTE : Temporary Validation Set == Test Set for Debugging
    val_indices = test_indices

    print(
        f"Debug] train, val, test : {len(train_indices)}, {len(val_indices)}, {len(test_indices)}"
    )

    train_data, val_data, test_data = [], [], []
    for i, adj in enumerate(adjs):
        if i in train_indices:
            train_data.append(adj)
        elif i in test_indices:
            test_data.append(adj)
        else:
            continue
        # NOTE : We do not use independent valid dataset
        if i in val_indices:
            val_data.append(adj)

    save_dir = DATA_PATH / args.dataset_name / args.original_data_dir / "raw"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_file = save_dir / f"train_spectre.pt"
    torch.save(train_data, save_file)
    save_file = save_dir / f"val_spectre.pt"
    torch.save(val_data, save_file)
    save_file = save_dir / f"test_spectre.pt"
    torch.save(test_data, save_file)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Ininitial data processing from pt file.")
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the pt file",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name for experiment",
    )
    parser.add_argument(
        "--original_data_dir",
        type=str,
        required=True,
        help="sub data dir for training argument",
    )
    parser.add_argument(
        "--allow_duplicates",
        action="store_true",
        help="Allow duplicated entries in csv file",
    )
    args = parser.parse_args()

    main()
