<p align="center">
  <h1 align="center">Discrete Diffusion Schrödinger Bridge Matching <br/> for Graph Transformation (DDSBM)</h1>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2410.01500-b31b1b.svg)](https://doi.org/10.48550/arXiv.2410.01500)
[![Python versions](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/downloads/)

Official implementation of **_Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation_** by Jun Hyeong Kim*, Seonghwan Kim*, Seokhyun Moon*, Hyeongwoo Kim*, Jeheon Woo*, Woo Youn Kim. [[arXiv](https://doi.org/10.48550/arXiv.2410.01500)][[OpenReview](https://openreview.net/forum?id=tQyh0gnfqW)]

We propose Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation (DDSBM)

# TODO (for code update)
- [ ] Add zinc raw data processing & initial pair matching codes
- [ ] Checkpoints update using Zenodo (as a `tar` file)
- [ ] Add polymer dataset, analysis codes
- [ ] Provide additional execution scripts
- [ ] Add brief summary of our work in README.md

## Environment installation
This code was tested with PyTorch 2.0.1, cuda 11.8 and torch_geometrics 2.3.1.

To install requirements, run:
```bash
git clone https://github.com/junhkim1226/DDSBM.git
cd DDSBM
conda create -c conda-forge -n ddsbm rdkit=2023.03.2 python=3.9
conda activate ddsbm
conda install -c conda-forge graph-tool=2.45
conda install -c "nvidia/label/cuda-11.8.0" cuda
pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install toolz dill==0.3.9 git+https://github.com/fabriziocosta/EDeN.git --no-deps # For NSPDK evaluation
pip install -e .
pre-commit install
g++ -O2 -std=c++11 -o ./src/ddsbm/analysis/orca/orca ./src/ddsbm/analysis/orca/orca.cpp # For unconditional generation (comm20, planar)
```

## Running the code
> [!IMPORTANT]
> Note that `${PROJECT_DIR}` refers to this directory.
The following section outlines the graph-to-graph transformation process. For details about other experiments, please refer to the [EXPERIMENTS.md](experiments/EXPERIMENTS.md).

### Data processing
> [!IMPORTANT]
> Before running the experiment, you should prepare the original data in `data` directory at the top of this repository.

To process the data, execute the command below:
```bash
python ./data/process_data.py \
    ${ORIGINAL_CSV_FILE} \
    --dataset_name ${DATASET_NAME} \
    --original_data_dir ${ORIGINAL_DATA_DIR}
```

<details>
<summary><strong>Example code for graph-to-graph generation</strong></summary>

```bash
# For pairs based on random-matching
ORIGINAL_CSV_FILE=./data/raw/ZINC250k_logp_2_4_random_matched_no_nH.csv
DATASET_NAME=zinc
ORIGINAL_DATA_DIR=ZINC250k_logp_2_4_random_matched_no_nH

python ./data/process_data.py \
    ${ORIGINAL_CSV_FILE} \
    --dataset_name ${DATASET_NAME} \
    --original_data_dir ${ORIGINAL_DATA_DIR}

# For pairs based on tanimoto similarity
ORIGINAL_CSV_FILE=./data/raw/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH.csv
DATASET_NAME=zinc
ORIGINAL_DATA_DIR=ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH

python ./data/process_data.py \
    ./data/raw/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH.csv \
    --dataset_name zinc \
    --original_data_dir ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH
```
</details>

Ensure that `ORIGINAL_CSV_FILE.csv` contains the columns `REF-SMI`(`X_0`) and `PRB-SMI`(`X_T`). In case there are duplicate SMILES entries, include the `--allow_duplicates` flag in the command.

Executing these commands will create the following directory structure:
```bash
data/
└── ${DATASET_NAME}
    └── ${ORIGINAL_DATA_DIR}
        ├── atom2weight.txt
        ├── max_mw.txt
        ├── max_num_atoms.txt
        └── raw
            ├── test.csv
            ├── train.csv
            └── val.csv
src/
└── ddsbm/
    ├── some
    ├── ...
    ├── python
    ├── ...
    └── files
├── main.py
...
```

### Training
After processing and splitting the data, run the following code to start the training process:

```bash
EXP=SB
DATASET=zinc
MIN_ALPHA=0.999
EPOCHS=300
NUM_GPUS_TO_USE=4

EXP_NAME=${EXP}_${MIN_ALPHA}

ddsbm-train \
    dataset.name=${DATASET} \
    general.name=${EXP_NAME} \
    general.gpus=${NUM_GPUS_TO_USE} \
    model.min_alpha=${MIN_ALPHA} \
    train.n_epochs=${EPOCHS} \
```

Upon execution, it creates copies of the preprocessed data into a directory named `${EXP_NAME}`.
If the `general.prepend_date_in_name` option is enabled, a new directory named `yyyy-mm-dd_${EXP_NAME}` is created instead.
This directory serves as the workspace where both training and sampling (for the SB iteration) are initiated.

Additionally, we provide two options for `graph_matching`, namely `general.skip_initial_graph_matching` and `general.skip_graph_matching`. If the former is enabled, the results from the negative log likelihood computation are saved to `${PROJECT_DIR}/data/zinc/${EXP_NAME}/processed/{train,test}_nll_df.csv`, while the latter makes `${PROJECT_DIR}/data/zinc/${EXP_NAME}/processed/train_nll_df_*.csv`.

For example,
```bash
data/zinc/2025-04-12_SB_0.999/
├── ...
├── processed
│   ├── train_nll_df.csv
│   ├── train_match_perm.pt
│   ├── match_config_train.yaml
│   ├── match_config_train_forward_5_last_seed42.yaml
│   ├── train_match_perm_forward_5_last_seed42.pt
│   ├── train_nll_df_forward_5_last_seed42.csv
│   └── ...
├── ...
└── raw
```

The SB training process generates multiple directories (named according to `${direction}_${iteration}`) containing checkpoints. Trained models for both `forward_n` and `backward_n` (with 0 ≤ n < SB_iteration) are saved in the output directory `${PROJECT_DIR}/outputs/zinc/${EXP_NAME}` as follows:


```bash
outputs/zinc/2025-04-12_SB_0.999/
├── backward_0
├── ...
├── forward_5
│   ├── chains
│   ├── checkpoints
│   ├── generated_joint_train_seed42_nfe100.pt
│   ├── graphs
│   └── wandb
└── main.log
```

If `general.prepend_date_in_name` is enabled, a directory named `yyyy-mm-dd_${EXP_NAME}` is created with subdirectories for each bridge direction and iteration number (e.g., `backward_0`, `forward_0`, `backward_1`, etc.). When each bridge training is completed, the results from the Markovian projection (sampling) are saved in that directory (e.g., `generated_joint_train_seed42_nfe100.pt`).

#### Training Bridge Model
The bridge model is identical to an SB model with `SB_iteration` set to 1. In our experiments, all results were configured with the `forward` direction. Additionally, to evaluate the bridge model and the SB model fairly, we set the training epochs for the bridge model to match the total epochs of the SB model `(outer_loop * n_epochs)`.

```bash
EXP=Bridge
DATASET=zinc
MIN_ALPHA=0.999
EPOCHS=3000  # NOTE: We should change epochs to compare bridge model with SB model
SAVE_EVERY=100  # NOTE: We should save checkpoints of Bridge model for every SB model's n_epoch (or its divisor)
NUM_GPUS_TO_USE=4

EXP_NAME=${EXP}_${MIN_ALPHA}

ddsbm-train \
    experiment=bridge \
    dataset.name=${DATASET} \
    general.name=${EXP_NAME} \
    general.gpus=${NUM_GPUS_TO_USE} \
    general.save_every_n_epochs=${SAVE_EVERY} \
    model.min_alpha=${MIN_ALPHA} \
    train.n_epochs=${EPOCHS} \
```

### Sampling
After the experiment, you can run the test (sampling) by executing ddsbm-test, which automatically loads the trained model's configuration.

```bash
DATASET_NAME=zinc
EXP_NAME=2025-04-12_SB_0.999
IDX=5
NUM_GPUS_TO_USE=4

ddsbm-test \
    general.test_only=./outputs/${DATASET_NAME}/${EXP_NAME}/forward_${IDX}/checkpoints/last.ckpt \
    general.gpus=${NUM_GPUS_TO_USE}
```

The sampled results are saved in the directory `${PROJECT_DIR}/outputs/test_forward_5_last`. Graph matching is then performed automatically, and the resulting NLL data file can be found in the data directory as `test_match_perm_forward_5_last_seed42.pt`.

```bash
outputs/zinc/2025-04-12_SB_0.999/
├── ...
└── test_forward_5_last
    ├── chains
    ├── graphs
    ├── generated_joint_test_seed42_nfe100.pt
    └── test_forward_5_last.log
```

#### Sampling Bridge Model
The bridge model is identical to an SB model with `SB_iteration` set to 1. In our experiments, all results were configured with the `forward` direction. Additionally, to evaluate the bridge model and the SB model fairly, we set the training epochs for the bridge model to match the total epochs of the SB model `(outer_loop * n_epochs)`.

```bash
DATASET_NAME=zinc
EXP_NAME=2025-04-12_Bridge_0.999
IDX=0  # NOTE: Bridge only has 0 as an IDX
EPOCH=1799 # NOTE: if SB model's outer iteration idx is 5, 300 * (5 + 1) - 1
NUM_GPUS_TO_USE=4

ddsbm-test \
    general.test_only="./outputs/${DATASET_NAME}/${EXP_NAME}/forward_${IDX}/checkpoints/epoch=${EPOCH}.ckpt" \
    general.gpus=${NUM_GPUS_TO_USE}
```

## Analysis
See [HERE](experiments/ANALYSIS.md).

## Error handling
To resolve an import error for graph_tools, use:

```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
```

For issues related to multi-GPU training and inference, which processes are stuck, set:

```
export NCCL_P2P_DISABLE=1
```

# Acknowledgement

This work is built upon [DiGress](https://github.com/cvignac/DiGress). Thanks to the authors.

# Citation
If you find this method and/or code useful, please consider citing

```{bibtex}
@article{kim2024discrete,
  title={Discrete Diffusion Schr$\backslash$" odinger Bridge Matching for Graph Transformation},
  author={Kim, Jun Hyeong and Kim, Seonghwan and Moon, Seokhyun and Kim, Hyeongwoo and Woo, Jeheon and Kim, Woo Youn},
  journal={arXiv preprint arXiv:2410.01500},
  year={2024}
}
```