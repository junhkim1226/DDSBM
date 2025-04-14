import argparse
import contextlib
import os
import pickle
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
import torch.multiprocessing as mp
from joblib import Parallel, delayed, parallel_backend
from omegaconf import OmegaConf
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolops

from ddsbm import DATA_PATH
from ddsbm.analysis.rdkit_functions import mol2smiles
from ddsbm.utils import PairData, PlaceHolder, to_dense

RDLogger.DisableLog("rdApp.error")


# fmt: off
ALLOWED_BONDS = {
    "H": 1, "C": 4, "N": 3, "O": 2, "F": 1, "B": 3, "Al": 3, "Si": 4, "P": [3, 5], "S": 4, "Cl": 1, "As": 3, "Br": 1, "I": 1, "Hg": [1, 2], "Bi": [3, 5], "Se": [2, 4, 6],
}
BOND_TYPE = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
# fmt: on


def generated_data_to_dense(data: PairData, max_num_atoms: int):
    """
    Extract dense version of {X,E}_{0,T}_p from PairData, which are generated data.

    Args:
        data: PairData
        max_num_atoms: int
    Returns:
        dense_data_0: PlaceHolder
        dense_data_T: PlaceHolder
        node_mask_0: torch.Tensor
        node_mask_T: torch.Tensor
    """

    x_0, edge_index_0, edge_attr_0, batch_0 = (
        data.x_0_p,
        data.edge_index_0_p,
        data.edge_attr_0_p.float(),
        None,
    )
    x_T, edge_index_T, edge_attr_T, batch_T = (
        data.x_T_p,
        data.edge_index_T_p,
        data.edge_attr_T_p.float(),
        None,
    )
    dense_data_0, node_mask_0 = to_dense(
        x_0, edge_index_0, edge_attr_0, batch_0, max_num_atoms
    )
    dense_data_T, node_mask_T = to_dense(
        x_T, edge_index_T, edge_attr_T, batch_T, max_num_atoms
    )
    return dense_data_0, dense_data_T, node_mask_0, node_mask_T


def original_data_to_dense(data: PairData, max_num_atoms: int):
    """
    Extract dense version of {X,E}_{0,T} from PairData, which are original data.

    Args:
        data: PairData
        max_num_atoms: int
    Returns:
        dense_data_0: PlaceHolder
        dense_data_T: PlaceHolder
        node_mask_0: torch.Tensor
        node_mask_T: torch.Tensor
    """

    x_0, edge_index_0, edge_attr_0, batch_0 = (
        data.x_0,
        data.edge_index_0,
        data.edge_attr_0.float(),
        None,
    )
    x_T, edge_index_T, edge_attr_T, batch_T = (
        data.x_T,
        data.edge_index_T,
        data.edge_attr_T.float(),
        None,
    )
    dense_data_0, node_mask_0 = to_dense(
        x_0, edge_index_0, edge_attr_0, batch_0, max_num_atoms
    )
    dense_data_T, node_mask_T = to_dense(
        x_T, edge_index_T, edge_attr_T, batch_T, max_num_atoms
    )
    return dense_data_0, dense_data_T, node_mask_0, node_mask_T


def build_molecule_with_atom_map(
    atom_types: torch.Tensor,
    edge_types: torch.Tensor,
    atom_decoder: List[str],
    atom_idx: List[int],
    verbose: bool = False,
) -> Chem.RWMol:
    """
    Builds a molecule from atom types and edge types.
    Returns an RWMol object, which needed Sanitization.

    Args:
        atom_types: torch.Tensor, shape (num_atoms,)
        edge_types: torch.Tensor, shape (num_atoms, num_atoms)
        atom_decoder: List[str]
        atom_idx: List[int]
        verbose: bool
    Returns:
        mol: Chem.RWMol
    """

    if verbose:
        print("building new molecule")
    edge_types = torch.where(edge_types < 5, edge_types, torch.zeros_like(edge_types))

    mol = Chem.RWMol()
    for idx, atom in enumerate(atom_types):
        a = Chem.Atom(atom_decoder[atom.item()])
        a.SetAtomMapNum(atom_idx[idx] + 1)
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types)
    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(
                bond[0].item(),
                bond[1].item(),
                BOND_TYPE[edge_types[bond[0], bond[1]].item()],
            )
            if verbose:
                print(
                    "bond added:",
                    bond[0].item(),
                    bond[1].item(),
                    edge_types[bond[0], bond[1]].item(),
                    BOND_TYPE[edge_types[bond[0], bond[1]].item()],
                )
    return mol


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence


def build_molecule_with_partial_charges(
    atom_types: torch.Tensor,
    edge_types: torch.Tensor,
    atom_decoder: List[str],
    verbose: bool = False,
    **kwargs,
):
    """
    Builds a molecule from atom types and edge types.
    Returns an RWMol object, which needed Sanitization.

    Args:
        atom_types: torch.Tensor, shape (num_atoms,)
        edge_types: torch.Tensor, shape (num_atoms, num_atoms)
        atom_decoder: List[str]
        atom_idx: List[int]
        verbose: bool
    Returns:
        mol: Chem.RWMol
    """

    if verbose:
        print("\nbuilding new molecule")
    edge_types = torch.where(edge_types < 5, edge_types, torch.zeros_like(edge_types))

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])
    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types)

    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(
                bond[0].item(),
                bond[1].item(),
                BOND_TYPE[edge_types[bond[0], bond[1]].item()],
            )
            if verbose:
                print(
                    "bond added:",
                    bond[0].item(),
                    bond[1].item(),
                    edge_types[bond[0], bond[1]].item(),
                    BOND_TYPE[edge_types[bond[0], bond[1]].item()],
                )
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if verbose:
                print("flag, valence", flag, atomid_valence)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if verbose:
                    print("atomic num of atom with a large valence", an)
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def delete_dummy_atoms(
    atom_types: torch.Tensor,
    edge_types: torch.Tensor,
    atom_decoder: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Deletes dummy atoms from atom types and edge types.

    Args:
        atom_types: torch.Tensor, shape (num_atoms,)
        edge_types: torch.Tensor, shape (num_atoms, num_atoms)
        atom_decoder: List[str]
    Returns:
        atom_types: torch.Tensor, shape (num_atoms,)
        edge_types: torch.Tensor, shape (num_atoms, num_atoms)
        atom_idx: List[int]
    """

    atom_idx = []
    if "X" in atom_decoder:
        # NOTE : graph-to-graph
        mask = atom_types != atom_decoder.index("X")
    else:
        # NOTE : unconditional (x_0 and x_T have the same # atom)
        mask = atom_types != -1
    atom_types = atom_types[mask]
    edge_types = edge_types[mask][:, mask]
    for idx, i in enumerate(mask):
        if i:
            atom_idx.append(idx)
    return atom_types, edge_types, atom_idx


def reset_atommap(mol: Chem.RWMol) -> Chem.RWMol:
    """
    Resets atom map numbers to 0.
    """

    for idx, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(0)
    return mol


def get_rwmol(
    dense_data: PlaceHolder,
    node_mask: torch.Tensor,
    atom_decoder: List[str],
    build_molecule_func: callable,
) -> Chem.RWMol:
    """
    Converts dense data to molecule.

    Args:
        dense_data: PlaceHolder
        node_mask: torch.Tensor
        atom_decoder: List[str]
    Returns:
        mol: Chem.RWMol
    """

    dense_data = dense_data.mask(node_mask, collapse=True)
    X = dense_data.X.squeeze(0)
    E = dense_data.E.squeeze(0)
    X, E, atom_idx = delete_dummy_atoms(X, E, atom_decoder)
    rwmol = build_molecule_func(
        atom_types=X,
        edge_types=E,
        atom_decoder=atom_decoder,
        verbose=False,
        atom_idx=atom_idx,
    )
    # rwmol = reset_atommap(rwmol)
    return rwmol


def parse_problems(problems):
    if len(problems) == 0:
        return []

    prb_list = []
    for problem in problems:
        prb_list.append(problem.Message())
    return prb_list


def check_mol_probs(rwmol: Chem.RWMol) -> Dict[str, float]:
    """
    Get QED and LOGP of the molecule.
    Can raise error when sanitization fails.

    Args:
        rwmol: Chem.RWMol
    Returns:
        Dict[str, float]
    """

    try:
        mol = rwmol.GetMol()
        Chem.SanitizeMol(mol)
    except Exception:
        problems = parse_problems(rdmolops.DetectChemistryProblems(rwmol))
        problems = " | ".join(problems)
        smi = Chem.MolToSmiles(rwmol)
    else:
        problems = None
        smi = Chem.MolToSmiles(mol)

    return dict(problems=problems, smiles=smi)


def analyze_generated_data(
    data: PairData,
    max_num_atoms: int,
    atom_decoder: List[str],
    direction: str,
) -> Dict[str, Union[float, Chem.RWMol]]:
    """
    Analyze the generated data.
    Compare original data and generated data in terms of NLL.
    Extract properties of the original molecules. (logP, QED)

    Args:
        Input: Tuple of the following
            - data: PairData
            - max_num_atoms: int
            - atom_decoder: List[str]
            - direction: str
    Returns:
    """
    RDLogger.DisableLog("rdApp.error")

    assert direction in ["forward", "backward"]
    (
        dense_data_0,
        dense_data_T,
        node_mask_0,
        node_mask_T,
    ) = original_data_to_dense(data, max_num_atoms)
    (
        dense_data_0_p,
        dense_data_T_p,
        node_mask_0_p,
        node_mask_T_p,
    ) = generated_data_to_dense(data, max_num_atoms)

    if direction == "forward":
        # NOTE: Compare (X_T, E_T) and (X_T_p, E_T_p)
        mol = get_rwmol(
            deepcopy(dense_data_T_p),
            node_mask_T_p.clone(),
            atom_decoder,
            build_molecule_with_atom_map,
        )
        relaxed_mol = get_rwmol(
            deepcopy(dense_data_T_p),
            node_mask_T_p.clone(),
            atom_decoder,
            build_molecule_with_partial_charges,
        )
    else:
        # NOTE: Compare (X_0, E_0) and (X_0_p, E_0_p)
        mol = get_rwmol(
            dense_data_0_p, node_mask_0_p, atom_decoder, build_molecule_with_atom_map
        )
        relaxed_mol = get_rwmol(
            dense_data_0_p,
            node_mask_0_p,
            atom_decoder,
            build_molecule_with_partial_charges,
        )

    result = {}
    result["generated_mol"] = mol
    result["generated_relaxed_mol"] = mol

    mol = reset_atommap(mol)
    smiles = mol2smiles(mol)
    if smiles is not None:
        try:
            mol_frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            smiles = mol2smiles(largest_mol)
        except Chem.rdchem.AtomValenceException:
            print("Valence error in GetmolFrags")
        except Chem.rdchem.KekulizeException:
            print("Can't kekulize molecule")

    relaxed_smiles = mol2smiles(relaxed_mol)
    if relaxed_smiles is not None:
        try:
            relaxed_mol_frags = Chem.rdmolops.GetMolFrags(
                relaxed_mol, asMols=True, sanitizeFrags=True
            )
            relaxed_largest_mol = max(
                relaxed_mol_frags, default=relaxed_mol, key=lambda m: m.GetNumAtoms()
            )
            relaxed_smiles = mol2smiles(relaxed_largest_mol)
        except Chem.rdchem.AtomValenceException:
            print("Valence error in GetmolFrags")
        except Chem.rdchem.KekulizeException:
            print("Can't kekulize molecule")

    # NOTE: Properties successfully calculated
    result["success"] = smiles is not None
    result["smiles"] = smiles
    result["relaxed_success"] = relaxed_smiles is not None
    result["relaxed_smiles"] = relaxed_smiles
    result["idx"] = data.idx.item()
    return result


def main():
    gen_data_path = args.gen_data_path.resolve()
    dataset_name = gen_data_path.parents[2].name
    exp_name = gen_data_path.parents[1].name
    direction = gen_data_path.parent.name.split("_")[1]
    data_path = DATA_PATH / dataset_name / exp_name
    assert data_path.exists(), f"Data path {data_path} does not exist"

    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        seed = gen_data_path.stem.split("_seed")[-1]
        if "_" in seed:
            seed = seed.split("_")[0]
        test_config_path = (
            gen_data_path.parent / ".hydra" / f"test_config_seed_{seed}.yaml"
        )
        test_config = OmegaConf.load(str(test_config_path))
        assert direction == test_config.train.bridge_direction
        dataset_config = test_config["dataset"]
        if dataset_config["unconditional"]:
            from ddsbm.datasets.uncond_jointmol_dataset import (
                JointMolDataModule,
                JointMolecularinfos,
                get_train_smiles,
            )

            datamodule = JointMolDataModule(test_config)
            dataset_infos = JointMolecularinfos(datamodule, test_config)
            train_smiles = get_train_smiles(
                test_config,
                datamodule.train_dataloader(),
                dataset_infos,
                evaluate_dataset=False,
            )

        else:
            from ddsbm.datasets.jointmol_dataset import (
                JointMolDataModule,
                JointMolecularinfos,
                get_train_smiles,
            )

            datamodule = JointMolDataModule(test_config)
            dataset_infos = JointMolecularinfos(datamodule, test_config)
            train_smiles = get_train_smiles(
                test_config,
                datamodule.train_dataloader(),
                dataset_infos,
                evaluate_dataset=False,
                direction=test_config.train.bridge_direction,
            )
    print(test_config["dataset"])
    print(dataset_config["unconditional"])

    result_path = gen_data_path.parent
    result_file = result_path / f"result_{direction}_seed{seed}.csv"
    print(f"RESULT FILE: {result_file}")
    print("EXISTS: ", result_file.exists())
    if result_file.exists():
        results = pd.read_csv(result_file)
    else:
        # NOTE: Some variables needed to convert mol from graph
        with (data_path / "max_num_atoms.txt").open("r") as f:
            max_num_nodes = int(f.read())
        with (data_path / "atom2weight.txt").open("r") as f:
            if dataset_config["unconditional"]:
                atom_decoder = [line.split()[0] for line in f]
            else:
                atom_decoder = ["X"] + [line.split()[0] for line in f]

        print("ATOM DECODER: ", atom_decoder)
        print("EXP NAME: ", exp_name)
        print("RESULT PATH: ", result_path)
        print(f"DIRECTION: {direction}")

        data = torch.load(gen_data_path, map_location="cpu")
        with parallel_backend("loky", n_jobs=args.num_workers):
            results = Parallel()(
                delayed(analyze_generated_data)(
                    pair_data, max_num_nodes, atom_decoder, direction
                )
                for pair_data in data.values()
            )

        results = pd.DataFrame(results)

        # NOTE: SAVE MOLS
        gen_mols = results["generated_mol"].values
        gen_idx = results["idx"].values
        idx2mol = dict(zip(gen_idx, gen_mols))
        with (result_path / f"gen_mols_seed{seed}.pkl").open("wb") as f:
            pickle.dump(idx2mol, f)

        gen_relaxed_mols = results["generated_relaxed_mol"].values
        idx2mol = dict(zip(gen_idx, gen_relaxed_mols))
        with (result_path / f"gen_relaxed_mols_seed{seed}.pkl").open("wb") as f:
            pickle.dump(idx2mol, f)

        results.drop(columns=["generated_mol"], inplace=True)
        results.drop(columns=["generated_relaxed_mol"], inplace=True)

        print(f"Saving generated csv for {gen_data_path} to {result_file}")
        results.to_csv(result_file)

    gens = results["smiles"].values[results["success"].values].tolist()
    uniq = len(set(gens)) / len(gens)
    nov = len(set(gens) - set(train_smiles)) / len(set(gens))

    relaxed_gens = (
        results["relaxed_smiles"].values[results["relaxed_success"].values].tolist()
    )
    relaxed_uniq = len(set(relaxed_gens)) / len(relaxed_gens)
    relaxed_nov = len(set(relaxed_gens) - set(train_smiles)) / len(set(relaxed_gens))
    with open(result_path / f"VUN_{direction}_{seed}.txt", "w") as f:
        f.write(f"Valid: {results['success'].mean()}\n")
        f.write(f"Unique: {uniq}\n")
        f.write(f"Novel: {nov}\n")
        f.write(f"Relaxed Valid: {results['relaxed_success'].mean()}\n")
        f.write(f"Relaxed Unique: {uniq}\n")
        f.write(f"Relaxed Novel: {nov}\n")
    return


if __name__ == "__main__":
    mp.set_sharing_strategy("file_system")
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gen_data_path",
        type=Path,
        help="Path to the generated files, which consists of list of `utils.PairData`",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for parallel processing. Default: 1",
    )
    args = parser.parse_args()

    main()
