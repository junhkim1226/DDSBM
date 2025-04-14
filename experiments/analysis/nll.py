import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ddsbm import DATA_PATH, RESULTS_PATH


def epoch_to_outer_iter(epoch: int) -> int:
    return (epoch + 1) // 300 - 1


def outer_iter_to_epoch(iteration: int) -> int:
    return (iteration + 1) * 300 - 1


def read_multiple_csvs(files: list[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    return df


def main():
    data_path = args.data_path.resolve() / "processed"
    exp_name = data_path.parent.name
    dataset_name = data_path.parents[1].name

    seeds = sorted(set(args.seeds))
    iterations = sorted(set(args.iterations))

    original_nll_file = data_path / "test_nll_df.csv"
    original_nll_df = pd.read_csv(original_nll_file)
    original_nll = original_nll_df["selected_nll"].values

    # Original
    original_nll_mean = np.mean(original_nll)

    nll_dic = defaultdict(list)
    # NOTE: We check nll results from various seeds for each iteration
    for iteration in iterations:
        sb_nll_files = [
            data_path / f"test_nll_df_{args.direction}_{iteration}_last_seed{seed}.csv"
            for seed in seeds
        ]
        sb_nll_files = sorted(
            sb_nll_files, key=lambda x: int(x.stem.split("_seed")[-1])
        )

        epoch = outer_iter_to_epoch(iteration)

        sb_nll_df = read_multiple_csvs(sb_nll_files)
        len_nlls = max(len(sb_nll_df), len(original_nll_df))

        sb_nll = sb_nll_df["selected_nll"].values
        nll_dic["nll"] += sb_nll.tolist()
        nll_dic["iteration"] += [iteration] * len_nlls

    nll_df = pd.DataFrame(nll_dic)
    mean_dic = defaultdict(list)
    mean_dic["iteration"].append("original")
    mean_dic["nll"].append(original_nll_mean)
    for iteration in iterations:
        partial_df = nll_df[nll_df["iteration"] == iteration]
        mean: pd.Series = partial_df["nll"].mean()
        mean_dic["iteration"].append(iteration)
        mean_dic["nll"].append(mean)

    seed_info = "_".join(list(map(str, seeds)))

    savedir = args.output_path.resolve()
    savedir.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(mean_dic)
    df = df.set_index("iteration")
    output_file = savedir / f"{dataset_name}-{exp_name}-{args.direction}-{seed_info}.csv"
    df.to_csv(output_file)
    print("SAVED NLL RESULT IN", output_file)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        help="An experiment's data path, which consists of various nll dfs "
        f"under {DATA_PATH}",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=RESULTS_PATH / "nll",
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
    args = parser.parse_args()

    main()
