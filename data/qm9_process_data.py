import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from ddsbm import DATA_PATH

SEED = 42
random.seed(SEED)


def split_data(df: pd.DataFrame, save_dir: Path):
    """
    Same As Digress Split
    """
    n_samples = len(df)

    n_train = 100000
    n_test = int(0.1 * n_samples)
    n_val = n_samples - (n_train + n_test)

    train_df, val_df, test_df = np.split(
        df.sample(frac=1, random_state=SEED), [n_train, n_val + n_train]
    )
    # NOTE : val == test
    val_df = test_df

    for csv_name in ["train", "val", "test"]:
        save_file = save_dir / f"{csv_name}.csv"
        save_df: pd.DataFrame
        if csv_name == "train":
            save_df = train_df
        elif csv_name == "val":
            save_df = val_df
        elif csv_name == "test":
            save_df = test_df
        try:
            # remove index column
            save_df = save_df.drop(columns=["Unnamed: 0"])
        except KeyError:
            pass
        save_df.to_csv(save_file)
    return


def main():
    """
    For Unconditional Generation, Drop lines related to "REF-SMI"
    """
    df = pd.read_csv(args.file)
    if not args.allow_duplicates:
        # Find duplicated
        # uniqueness check
        # assert df.shape == df.drop_duplicates(["REF-SMI"]).shape
        assert df.shape == df.drop_duplicates(["PRB-SMI"]).shape

    atom_types = {}
    max_mw = 0
    max_num_atoms = 0
    periodictable = Chem.GetPeriodicTable()

    for idx, row in df.iterrows():
        # ref_mol = Chem.MolFromSmiles(row["REF-SMI"])
        prb_mol = Chem.MolFromSmiles(row["PRB-SMI"])
        # ref_mw = Descriptors.MolWt(ref_mol)
        prb_mw = Descriptors.MolWt(prb_mol)
        # max_mw = max(max_mw, ref_mw, prb_mw)
        max_mw = max(max_mw, prb_mw)
        # ref_num_atoms = ref_mol.GetNumAtoms()
        prb_num_atoms = prb_mol.GetNumAtoms()
        # max_num_atoms = max(max_num_atoms, ref_num_atoms, prb_num_atoms)
        max_num_atoms = max(max_num_atoms, prb_num_atoms)
        for atom in prb_mol.GetAtoms():
            atom_types[atom.GetSymbol()] = atom.GetAtomicNum()

    atom2weight = {
        atom: periodictable.GetAtomicWeight(atom) for atom, idx in atom_types.items()
    }
    print("File: ", args.file.resolve())
    print(
        "Atom types: ",
        [key for key, value in sorted(atom2weight.items(), key=lambda x: x[1])],
    )
    print(
        "Atom weights: ",
        [value for key, value in sorted(atom2weight.items(), key=lambda x: x[1])],
    )
    print("Max num atoms: ", max_num_atoms)
    print("Max MW: ", max_mw)

    save_dir = DATA_PATH / args.dataset_name / args.sub_data_dir / "raw"
    save_dir.mkdir(parents=True, exist_ok=True)

    split_data(df, save_dir)
    with open(save_dir.parent / "atom2weight.txt", "w") as f:
        for atom, weight in sorted(atom2weight.items(), key=lambda x: x[1]):
            f.write(f"{atom} {weight}\n")
    with open(save_dir.parent / "max_num_atoms.txt", "w") as f:
        f.write(f"{max_num_atoms}\n")
    with open(save_dir.parent / "max_mw.txt", "w") as f:
        f.write(f"{max_mw}\n")
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Ininitial data processing from csv file.")
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the csv file",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name for experiment",
    )
    parser.add_argument(
        "--sub_data_dir",
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
