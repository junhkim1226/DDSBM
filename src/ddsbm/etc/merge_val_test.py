from pathlib import Path
from shutil import copy2, copytree, move

import pandas as pd
import torch
from torch_geometric.data import separate

from ddsbm.utils import _collate


def merge_original_valtest_to_processed_test(data_path: Path, regex: str):
    for dir in data_path.glob(regex):
        valid_data_path = dir / "811_valid_data.pt"
        valid_data = torch.load(valid_data_path)
        for key, val in valid_data.items():
            assert key == val.idx

        test_data_path = dir / "811_test_data.pt"
        test_data = torch.load(test_data_path)
        for key, val in test_data.items():
            assert key == val.idx

        dic = {}
        dic.update(valid_data)
        dic.update(test_data)
        sorted_dic = dict(sorted(dic.items()))
        data_list = list(sorted_dic.values())

        new_test_collated_data = _collate(data_list)
        assert torch.all(
            torch.argsort(new_test_collated_data[0].idx)
            == torch.arange(len(new_test_collated_data[0].idx))
        )

        _td = torch.load(
            "/home/mseok/work/DL/D3BM/data/zinc/2024-09-23_debug_T_0.99795_marginal/test_data.pt"
        )
        for key in _td.keys():
            val = sorted_dic[key]
            _val = _td[key]
            assert torch.all(val.x_0_p == _val.x_0_p)
            assert torch.all(val.edge_index_0_p == _val.edge_index_0_p)
            assert torch.all(val.edge_attr_0_p == _val.edge_attr_0_p)

        assert len(sorted_dic) == len(valid_data) + len(test_data)
        processed_test_data_path = dir / "processed" / "test_data.pt"
        print("Overwriting processed test data: ", processed_test_data_path)
        torch.save(new_test_collated_data, processed_test_data_path)
        test_data_path = dir / "test_data.pt"
        print("Overwriting test data: ", test_data_path)
        torch.save(sorted_dic, test_data_path)
    return


def merge_valtest(data_path: Path, regex: str):
    for dir in data_path.glob(regex):
        print("DATA DIR: ", dir)
        valid_csv_path = dir / "raw" / "val_zinc.csv"
        test_csv_path = dir / "raw" / "test_zinc.csv"
        valid_df = pd.read_csv(valid_csv_path)
        test_df = pd.read_csv(test_csv_path)
        new_test_df = pd.concat([valid_df, test_df])
        # save "Unnamed: 0" column as index column
        new_test_df = new_test_df.set_index("Unnamed: 0")
        print("Overwriting raw test csv: ", test_csv_path)
        new_test_df.to_csv(test_csv_path, index_label="")

        valid_data_path = dir / "valid_data.pt"
        valid_data = torch.load(valid_data_path)
        for key, val in valid_data.items():
            assert key == val.idx
        test_data_path = dir / "test_data.pt"
        test_data = torch.load(test_data_path)
        for key, val in test_data.items():
            assert key == val.idx

        dic = {}
        dic.update(valid_data)
        dic.update(test_data)
        sorted_dic = dict(sorted(dic.items()))
        data_list = list(sorted_dic.values())
        new_test_collated_data = _collate(data_list)
        assert torch.all(
            torch.argsort(new_test_collated_data[0].idx)
            == torch.arange(len(new_test_collated_data[0].idx))
        )

        assert len(sorted_dic) == len(valid_data) + len(test_data)
        processed_test_data_path = dir / "processed" / "test_data.pt"
        print("Overwriting processed test data: ", processed_test_data_path)
        torch.save(new_test_collated_data, processed_test_data_path)
        test_data_path = dir / "test_data.pt"
        print("Overwriting test data: ", test_data_path)
        torch.save(sorted_dic, test_data_path)
    return


def copy_data_and_merge_valtest(data_path: Path, regex: str):
    for dir in data_path.glob(regex):
        print("DATA DIR: ", dir)

        print("Copying data...")
        copytree(dir / "processed", dir / "811_processed")
        copytree(dir / "raw", dir / "811_raw")
        copy2(dir / "valid_data.pt", dir / "811_valid_data.pt")
        copy2(dir / "test_data.pt", dir / "811_test_data.pt")

        for file in (dir / "processed").glob("match_config_test*.yaml"):
            file.unlink()
        for file in (dir / "processed").glob("test_match_perm*.pt"):
            file.unlink()
        for file in (dir / "processed").glob("test_nll_df*.csv"):
            file.unlink()

        valid_csv_path = dir / "raw" / "val_zinc.csv"
        test_csv_path = dir / "raw" / "test_zinc.csv"
        valid_df = pd.read_csv(valid_csv_path)
        test_df = pd.read_csv(test_csv_path)
        new_test_df = pd.concat([valid_df, test_df])
        # save "Unnamed: 0" column as index column
        new_test_df = new_test_df.set_index("Unnamed: 0")
        print("Overwriting raw test csv: ", test_csv_path)
        new_test_df.to_csv(test_csv_path, index_label="")

        valid_data_path = dir / "811_valid_data.pt"
        valid_data = torch.load(valid_data_path)
        for key, val in valid_data.items():
            assert key == val.idx

        test_data_path = dir / "811_test_data.pt"
        test_data = torch.load(test_data_path)
        for key, val in test_data.items():
            assert key == val.idx

        dic = {}
        dic.update(valid_data)
        dic.update(test_data)
        sorted_dic = dict(sorted(dic.items()))
        data_list = list(sorted_dic.values())

        new_test_collated_data = _collate(data_list)
        assert torch.all(
            torch.argsort(new_test_collated_data[0].idx)
            == torch.arange(len(new_test_collated_data[0].idx))
        )

        assert len(sorted_dic) == valid_df.shape[0] + test_df.shape[0]
        assert len(sorted_dic) == len(valid_data) + len(test_data)
        processed_test_data_path = dir / "processed" / "test_data.pt"
        print("Overwriting processed test data: ", processed_test_data_path)
        torch.save(new_test_collated_data, processed_test_data_path)
        test_data_path = dir / "test_data.pt"
        print("Overwriting test data: ", test_data_path)
        torch.save(sorted_dic, test_data_path)
    return


def mv_outputs(output_path: Path, regex: str):
    for dir in output_path.glob(regex):
        for test_dir in dir.glob("test_*"):
            print("TEST DIR: ", test_dir)
            print("MOVING TO: ", test_dir.parent / f"811_{test_dir.name}")
            move(test_dir, test_dir.parent / f"811_{test_dir.name}")
    return


def main():
    # merge_valtest(DATA_PATH, "2024-09-27_SB_*")
    # merge_valtest(DATA_PATH, "2024-09-27_Bridge_R_0.999_uniform")
    # merge_valtest(DATA_PATH, "2024-09-29_Bridge_*")
    # merge_valtest(DATA_PATH, "2024-11-13_*")
    # merge_valtest(DATA_PATH, "2024-11-14_*")
    # merge_valtest(DATA_PATH, "2024-11-17_*")
    merge_valtest(DATA_PATH, "2024-11-21_*")

    # copy_data_and_merge_valtest(DATA_PATH, "2024-08-30_*")
    # copy_data_and_merge_valtest(DATA_PATH, "2024-09-0?_*")
    # mv_outputs(OUTPUT_PATH, "2024-08-30_*")
    # mv_outputs(OUTPUT_PATH, "2024-09-0?_*")
    # copy_data_and_merge_valtest(DATA_PATH, "2024-09-23_R_*")
    # copy_data_and_merge_valtest(DATA_PATH, "2024-09-23_T_*")
    # mv_outputs(OUTPUT_PATH, "2024-09-23_R_*")
    # mv_outputs(OUTPUT_PATH, "2024-09-23_T_*")
    return


if __name__ == "__main__":
    DATA_PATH = Path("/home/mseok/work/DL/D3BM/data/zinc")
    OUTPUT_PATH = Path("/home/mseok/work/DL/D3BM/outputs/SB/zinc")

    main()
