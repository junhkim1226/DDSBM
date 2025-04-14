from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

RANDOM_RAW_PATH = Path(
    "/home/mseok/work/DL/D3BM/data/zinc/ZINC250k_logp_2_4_random_matched_no_nH/raw"
)
TANIMOTO_RAW_PATH = Path(
    "/home/mseok/work/DL/D3BM/data/zinc/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH/raw"
)
OUTPUT_DATA_PATH = Path("/home/mseok/work/DL/D3BM/data/zinc")
SB_DATA_PATH = Path("/home/mseok/work/DL/D3BM/data/zinc/2024-09-27_SB_R_0.999_uniform")
MAX_NUM_ATOMS = int(open(SB_DATA_PATH / "max_num_atoms.txt").read())
assert MAX_NUM_ATOMS == 37  # ZINC
G2G_EXP_PATH = Path("/home/share/DATA/junhkim_DDSBM/240926_g2g")

original_df_dic = {}
original_df_dic["random_val"] = pd.read_csv(RANDOM_RAW_PATH / "val_zinc.csv")
original_df_dic["random_test"] = pd.read_csv(RANDOM_RAW_PATH / "test_zinc.csv")
original_df_dic["tanimoto_val"] = pd.read_csv(TANIMOTO_RAW_PATH / "val_zinc.csv")
original_df_dic["tanimoto_test"] = pd.read_csv(TANIMOTO_RAW_PATH / "test_zinc.csv")


def gather_data(exp_name):
    input_dic = defaultdict(list)
    result_dic = defaultdict(list)
    natom_success_dic = defaultdict(list)
    random_data_dic = defaultdict(list)
    tanimoto_data_dic = defaultdict(list)

    idx_df = original_df_dic["random_val"]
    indice = idx_df["Unnamed: 0"].values

    with open(G2G_EXP_PATH / exp_name / "random_valid_results.txt") as f:
        for line_idx, line in enumerate(f):
            line = line.strip().split()
            input_dic[f"{exp_name}_random_val"].append(line[0])
            result_dic[f"{exp_name}_random_val"].append(line[1])
            natom_success = Chem.MolFromSmiles(line[1]).GetNumAtoms() <= MAX_NUM_ATOMS
            natom_success_dic[f"{exp_name}_random_val"].append(natom_success)
            if natom_success:
                random_data_dic["REF-SMI"].append(line[0])
                random_data_dic["PRB-SMI"].append(line[1])
                random_data_dic["idx"].append(indice[line_idx])

    with open(G2G_EXP_PATH / exp_name / "random_test_results.txt") as f:
        for line_idx, line in enumerate(f):
            line = line.strip().split()
            input_dic[f"{exp_name}_random_test"].append(line[0])
            result_dic[f"{exp_name}_random_test"].append(line[1])
            natom_success = Chem.MolFromSmiles(line[1]).GetNumAtoms() <= MAX_NUM_ATOMS
            natom_success_dic[f"{exp_name}_random_test"].append(natom_success)
            if natom_success:
                random_data_dic["REF-SMI"].append(line[0])
                random_data_dic["PRB-SMI"].append(line[1])
                random_data_dic["idx"].append(indice[line_idx])

    with open(G2G_EXP_PATH / exp_name / "tanimoto_valid_results.txt") as f:
        for line_idx, line in enumerate(f):
            line = line.strip().split()
            input_dic[f"{exp_name}_tanimoto_val"].append(line[0])
            result_dic[f"{exp_name}_tanimoto_val"].append(line[1])
            natom_success = Chem.MolFromSmiles(line[1]).GetNumAtoms() <= MAX_NUM_ATOMS
            natom_success_dic[f"{exp_name}_tanimoto_val"].append(natom_success)
            if natom_success:
                tanimoto_data_dic["REF-SMI"].append(line[0])
                tanimoto_data_dic["PRB-SMI"].append(line[1])
                tanimoto_data_dic["idx"].append(indice[line_idx])

    with open(G2G_EXP_PATH / exp_name / "tanimoto_test_results.txt") as f:
        for line_idx, line in enumerate(f):
            line = line.strip().split()
            input_dic[f"{exp_name}_tanimoto_test"].append(line[0])
            result_dic[f"{exp_name}_tanimoto_test"].append(line[1])
            natom_success = Chem.MolFromSmiles(line[1]).GetNumAtoms() <= MAX_NUM_ATOMS
            natom_success_dic[f"{exp_name}_tanimoto_test"].append(natom_success)
            if natom_success:
                tanimoto_data_dic["REF-SMI"].append(line[0])
                tanimoto_data_dic["PRB-SMI"].append(line[1])
                tanimoto_data_dic["idx"].append(indice[line_idx])

    # NOTE: check if the input smiles are matched with the original ones
    for key, value in input_dic.items():
        _key = "_".join(key.split("_")[1:])
        df = original_df_dic[_key]
        assert np.all(df["REF-SMI"].values == np.array(value)), f"{key} is not matched"

    random_output_path = OUTPUT_DATA_PATH / f"{exp_name}_random" / "raw"
    random_output_path.mkdir(exist_ok=True, parents=True)
    random_data_df = pd.DataFrame(random_data_dic)
    random_data_df.to_csv(random_output_path / "train_zinc.csv")
    random_data_df.to_csv(random_output_path / "val_zinc.csv")
    random_data_df.to_csv(random_output_path / "test_zinc.csv")

    tanimoto_output_path = OUTPUT_DATA_PATH / f"{exp_name}_tanimoto_sim" / "raw"
    tanimoto_output_path.mkdir(exist_ok=True, parents=True)
    tanimoto_data_df = pd.DataFrame(tanimoto_data_dic)
    tanimoto_data_df.to_csv(tanimoto_output_path / "train_zinc.csv")
    tanimoto_data_df.to_csv(tanimoto_output_path / "val_zinc.csv")
    tanimoto_data_df.to_csv(tanimoto_output_path / "test_zinc.csv")

    random_full_result = defaultdict(list)
    random_partial_result = defaultdict(list)
    tanimoto_full_result = defaultdict(list)
    tanimoto_partial_result = defaultdict(list)
    for key, value in result_dic.items():
        _key = "_".join(key.split("_")[1:])

        if "random" in key:
            full_result = random_full_result
            partial_result = random_partial_result
        else:
            full_result = tanimoto_full_result
            partial_result = tanimoto_partial_result

        df = original_df_dic[_key]
        success = [v is not None for v in value]
        natom_success = np.array(natom_success_dic[key])
        idx = df["Unnamed: 0"].values

        full_result["success"].extend(success)
        full_result["smiles"].extend(value)
        full_result["idx"].extend(idx.tolist())

        partial_result["success"].extend((np.array(success)[natom_success]).tolist())
        partial_result["smiles"].extend((np.array(value)[natom_success]).tolist())
        partial_result["idx"].extend(idx[natom_success].tolist())

    random_df = pd.DataFrame(random_full_result)
    random_dir = G2G_EXP_PATH / exp_name / "random"
    random_dir.mkdir(exist_ok=True)
    random_df.to_csv(random_dir / "result_full_seed42.csv")

    tanimoto_df = pd.DataFrame(tanimoto_full_result)
    tanimoto_dir = G2G_EXP_PATH / exp_name / "tanimoto_sim"
    tanimoto_dir.mkdir(exist_ok=True)
    tanimoto_df.to_csv(tanimoto_dir / "result_full_seed42.csv")

    random_df = pd.DataFrame(random_partial_result)
    random_dir = G2G_EXP_PATH / exp_name / "random"
    random_dir.mkdir(exist_ok=True)
    random_df.to_csv(random_dir / "result_seed42.csv")

    tanimoto_df = pd.DataFrame(tanimoto_partial_result)
    tanimoto_dir = G2G_EXP_PATH / exp_name / "tanimoto_sim"
    tanimoto_dir.mkdir(exist_ok=True)
    tanimoto_df.to_csv(tanimoto_dir / "result_seed42.csv")
    return


def main():
    gather_data("atomg2g")
    gather_data("hierg2g")
    return


if __name__ == "__main__":
    main()
