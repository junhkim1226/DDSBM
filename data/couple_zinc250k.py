import random
from pathlib import Path

import numpy as np
import pandas as pd
import pygmtools as pygm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from ddsbm import DATA_PATH


def get_fp_list(mol_list: list) -> list:
    """Get Morgan fingerprint of a list of molecules

    Args:
        mol_list (list): List of RDKit molecule objects

    Returns:
        list: List of Morgan fingerprints
    """
    fps = []
    for mol in mol_list:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fps.append(fp)
    return fps


def get_scaffold_list(mol_list: list) -> list:
    """Get the scaffold of a list of molecules

    Args:
        mol_list (list): List of RDKit molecule objects

    Returns:
        list: List of RDKit molecule objects representing the scaffold
    """
    scaffolds = [MurckoScaffold.GetScaffoldForMol(mol) for mol in mol_list]
    return scaffolds


def get_similarity_matrix(mols_0: list, mols_T: list) -> np.ndarray:
    """Get similarity matrix between two sets of fingerprints

    Args:
        fps_0 (list): List of fingerprints
        fps_T (list): List of fingerprints

    Returns:
        np.ndarray: Similarity matrix
    """
    fps_0 = get_fp_list(mols_0)
    fps_T = get_fp_list(mols_T)
    sim_matrix = np.zeros((len(fps_0), len(fps_T)))
    for n in range(len(fps_0)):
        s = DataStructs.BulkTanimotoSimilarity(fps_0[n], fps_T)
        s = np.array(s)
        sim_matrix[n] = s
    return sim_matrix


def similarity_coupling(smi_list_0, smi_list_T, split_num):
    smi_list_0 = np.array(smi_list_0)
    smi_list_T = np.array(smi_list_T)
    length = len(smi_list_0)
    output_0, output_T = [], []
    for i in range(split_num):
        print(f"Debug] Iteration with {i}th split...")
        smi_list_0_split = smi_list_0[
            int(i * length / split_num) : int((i + 1) * length / split_num)
        ]
        smi_list_T_split = smi_list_T[
            int(i * length / split_num) : int((i + 1) * length / split_num)
        ]
        mol_list_0 = [Chem.MolFromSmiles(smi) for smi in smi_list_0_split]
        mol_list_T = [Chem.MolFromSmiles(smi) for smi in smi_list_T_split]
        print("Debug] Build Similarity Matrix...")
        sim_matrix = get_similarity_matrix(mol_list_0, mol_list_T)
        print("Debug] Hungarian Matching...")
        match_matrix = pygm.hungarian(sim_matrix)
        perm = match_matrix.argmax(axis=1)
        output_0 += smi_list_0_split.tolist()
        output_T += smi_list_T_split[perm].tolist()
    return output_0, output_T


def get_similarity(a: Chem.rdchem.Mol, b: Chem.rdchem.Mol):
    fp1 = AllChem.GetMorganFingerprintAsBitVect(a, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(b, 2, nBits=2048, useChirality=False)

    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    return sim


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target_property", type=str)
    args = parser.parse_args()

    # save directory
    save_dir = DATA_PATH / "raw"

    # configuration
    seed = args.seed
    target_prop = args.target_property
    # percentage = args.percentage
    # print(f"Seed: {seed}, Target Property: {target_prop}, Percentage: {percentage}%")
    print(f"Seed: {seed}, Target Property: {target_prop}")

    # set the seed
    random.seed(seed)
    np.random.seed(seed)

    # read the data
    data1 = pd.read_csv("./zinc250k_logp_retrieved1.csv")
    data2 = pd.read_csv("./zinc250k_logp_retrieved2.csv")

    data1_smis = data1["SMILES"].tolist()
    data2_smis = data2["SMILES"].tolist()

    num_data = min(len(data1_smis), len(data2_smis))
    data1_smis = data1_smis[:num_data]
    data2_smis = data2_smis[:num_data]
    print(f"Data1: {len(data1_smis)}, Data2: {len(data2_smis)}")

    # similarity coupling
    coupled_data1, coupled_data2 = similarity_coupling(data1_smis, data2_smis, 1)

    # save the data
    data_to_save = {
        "REF-SMI": [],
        "PRB-SMI": [],
        "REF-TPSA": [],
        "PRB-TPSA": [],
        "REF-QED": [],
        "PRB-QED": [],
        "REF-SAS": [],
        "PRB-SAS": [],
        "REF-PLOGP": [],
        "PRB-PLOGP": [],
        "TANIMOTO-SIM": [],
    }

    for smi1, smi2 in zip(coupled_data1, coupled_data2):
        mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
        row1, row2 = (
            data1[data1["SMILES"] == smi1],
            data2[data2["SMILES"] == smi2],
        )
        data_to_save["REF-SMI"].append(smi1)
        data_to_save["PRB-SMI"].append(smi2)

        for prop in ["TPSA", "QED", "SAS", "PLOGP"]:
            data_to_save[f"REF-{prop}"].append(row1[prop].values[0])
            data_to_save[f"PRB-{prop}"].append(row2[prop].values[0])

        sim = get_similarity(mol1, mol2)
        data_to_save["TANIMOTO-SIM"].append(sim)

    data_df = pd.DataFrame(data_to_save)
    data_df.to_csv(
        save_dir.joinpath(
            "ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH.csv"
        )
    )
