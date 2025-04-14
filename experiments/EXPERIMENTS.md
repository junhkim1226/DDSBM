# Running Experiments

> [!NOTE]
> One can see the definition about each variable in the following explanation in [main README.md](../README.md).

## Graph-to-graph Transformation
In our code, we provide two graph-to-graph transformation ([zinc](#zinc), [polymer](#polymer)).

### ZINC
#### Data processing
```{bash}
ORIGINAL_CSV_FILE=./data/raw/ZINC250k_logp_2_4_random_matched_no_nH.csv
DATASET_NAME=zinc
ORIGINAL_DATA_DIR=ZINC250k_logp_2_4_random_matched_no_nH

python ${PROJECT_DIR}/data/process_data.py \
    ${ORIGINAL_CSV_FILE} \
    --dataset_name ${DATASET_NAME} \
    --original_data_dir ${ORIGINAL_DATA_DIR}
```

#### Training
```{bash}
DATASET_NAME=zinc
EXP_NAME=SB_0.999
NUM_GPUS_TO_USE=4
ORIGINAL_DATA_DIR=ZINC250k_logp_2_4_random_matched_no_nH

ddsbm-train \
    dataset.name=${DATASET} \
    dataset.original_data_dir=${ORIGINAL_DATA_DIR} \
    general.name=${EXP_NAME} \
    general.gpus=${NUM_GPUS_TO_USE}
```

#### Sampling

To sampling with our code, you can just use the checkpoint file.
```{bash}
NUM_GPUS_TO_USE=4
SEED=42  # generation seed

ddsbm-test \
    general.test_only=../outputs/zinc/2025-04-06_SB_0.999/forward_5/checkpoints/last.ckpt \
    general.gpus=${NUM_GPUS_TO_USE} \
    general.seed=${SEED}
```

This will generate resulting graph data as tensor, with the name of `../outputs/zinc/2025-04-06_SB_0.999/test_forward_5_last/generated_joint_test_42_nfe100.pt`.

#### Analysis
For analysis of zinc graph-to-graph generation, please refer to [ANALYSIS.md](./ANALYSIS.md).

---

### Polymer
#### Data processing
```{bash}
```

#### Training
```{bash}
```

#### Sampling
```{bash}
```

#### Analysis
For analysis of polymer graph-to-graph generation, please refer to [ANALYSIS.md](./ANALYSIS.md).

---

## Unconditional Generation
In our code, we provide three unconditional generation ([qm9](#qm9), [community-20](#community-20), [planar](#planar)).

### QM9
> [!NOTE]
> Data processing parts should be done in [PROJECT_DIR](..).

#### Data processing
Similar to graph-to-graph generation, you can preprocess raw qm9 dataset as follow:

```{bash}
python data/qm9_process_data.py \
    data/raw/qm9_processed.csv \
    --dataset_name uncond_qm9 \
    --original_data_dir qm9
```

This will make the following data directory:
```{bash}
data/uncond_qm9/qm9/
├── atom2weight.txt
├── max_mw.txt
├── max_num_atoms.txt
└── raw
    ├── train.csv
    ├── val.csv
    └── test.csv
```

#### Training
Then, you can run experiments using follow command:
```{bash}
DATASET_NAME=uncond_qm9
EXP_NAME=SB_QM9_marginal
NUM_GPUS_TO_USE=4

ddsbm-train \
  dataset=${DATASET_NAME} \
  general.name=${EXP_NAME} \
  general.check_val_every_n_epochs=50 \
  general.sample_every_val=2 \
  general.gpus=${NUM_GPUS_TO_USE} \
  graph_match.only_post_process=false \
  experiment.outer_loops=2 \
  experiment.skip_data_generation=false \
  experiment.skip_initial_graph_matching=false \
  experiment.skip_graph_matching=false \
  model.hidden_mlp_dims.X=256 \
  model.hidden_mlp_dims.E=128 \
  model.hidden_mlp_dims.y=128 \
  model.hidden_dims.dx=256 \
  model.hidden_dims.de=64 \
  model.hidden_dims.dy=64 \
  model.hidden_dims.dim_ffX=256 \
  model.hidden_dims.dim_ffE=128 \
  model.hidden_dims.dim_ffy=128 \
  model.hidden_dims.n_head=8 \
  model.diffusion_steps=100 \
  model.min_alpha=0.999 \
  model.n_layers=9 \
  model.transition='marginal' \
  train.lr=0.0004 \
  train.n_epochs=100 \
  train.batch_size=512
```

#### Sampling
After the experiment, you can run test (sampling) by executing `ddsbm-test`
This automatically load trained model's configuration.

```{bash}
DATASET_NAME=uncond_qm9
EXP_NAME=2025-04-06_SB_QM9_marginal
IDX=1
NUM_GPUS_TO_USE=4

ddsbm-test \
    dataset=$DATASET_NAME \
    general.test_only=./outputs/${DATASET_NAME}/${EXP_NAME}/forward_${IDX}/checkpoints/last.ckpt \
    general.gpus=${NUM_GPUS_TO_USE}

```

#### Analysis
Analysis of QM9 is same with graph-to-graph generation. Please refer to [ANALYSIS.md](./ANALYSIS.md).

---

### Community-20
#### Data processing
You can preprocess raw community-20 dataset as follow:
```{bash}
python data/spectre_process_data.py \
  'data/raw/community_12_21_100.pt' \
  --dataset_name 'uncond_comm20' \
  --original_data_dir 'comm20'
```

This will make the following data directory:
```{bash}
./data/uncond_comm20/comm20/
└── raw
    ├── train_spectre.pt
    ├── val_spectre.pt
    └── test_spectre.pt
```

#### Training
Then, you can run experiments using follow command:
```{bash}
DATASET_NAME=uncond_comm20
EXP_NAME=SB_comm20_marginal
NUM_GPUS_TO_USE=4

ddsbm-train \
  dataset=${DATASET_NAME} \
  general.name=${EXP_NAME} \
  general.gpus=${NUM_GPUS_TO_USE} \
  general.check_val_every_n_epochs=5000 \
  general.sample_every_val=2 \
  graph_match.only_post_process=false \
  experiment.outer_loops=5 \
  experiment.skip_data_generation=false \
  experiment.skip_initial_graph_matching=false \
  experiment.skip_graph_matching=false \
  model.hidden_mlp_dims.X=256 \
  model.hidden_mlp_dims.E=128 \
  model.hidden_mlp_dims.y=128 \
  model.hidden_dims.dx=256 \
  model.hidden_dims.de=64 \
  model.hidden_dims.dy=64 \
  model.hidden_dims.dim_ffX=256 \
  model.hidden_dims.dim_ffE=128 \
  model.hidden_dims.dim_ffy=128 \
  model.hidden_dims.n_head=8 \
  model.diffusion_steps=500 \
  model.min_alpha=0.9998 \
  model.n_layers=8 \
  model.transition='marginal' \
  train.lr=0.0002 \
  train.n_epochs=200000 \
  train.batch_size=20 \
```

#### Sampling
After the experiment, you can run test (sampling) by executing `ddsbm-test`.
This automatically load trained model's configuration.

```{bash}
DATASET_NAME=uncond_comm20
EXP_NAME=2025-04-06_SB_comm20_marginal
IDX=4
NUM_GPUS_TO_USE=4

ddsbm-test \
    dataset=$DATASET_NAME \
    general.test_only=./outputs/${DATASET_NAME}/${EXP_NAME}/forward_${IDX}/checkpoints/last.ckpt \
    general.gpus=${NUM_GPUS_TO_USE}

```

This code not only sample new graph from prior distribution, but also measure sampling metric (degree, orbit, clsuter).

#### Analysis
The sampling results and metrics can be obtained from `stdout`.

---

### Planar
#### Data processing
You can preprocess raw planar dataset as follow:
```{bash}
python data/spectre_process_data.py \
  'data/raw/planar_64_200.pt' \
  --dataset_name 'uncond_planar' \
  --original_data_dir 'planar'
```

This will make the following data directory:
```{bash}
./data/uncond_planar/planar/
└── raw
    ├── train_spectre.pt
    ├── val_spectre.pt
    └── test_spectre.pt
```

#### Training
Then, you can run experiments using follow command:
```{bash}
DATASET_NAME=uncond_planar
EXP_NAME=SB_planar_uniform
NUM_GPUS_TO_USE=2

ddsbm-train \
  dataset=${DATASET_NAME} \
  general.name=${EXP_NAME} \
  general.check_val_every_n_epochs=1000 \
  general.gpus=${NUM_GPUS_TO_USE} \
  graph_match.only_post_process=false \
  experiment.outer_loops=10 \
  experiment.skip_data_generation=false \
  experiment.skip_initial_graph_matching=false \
  experiment.skip_graph_matching=false \
  model.hidden_mlp_dims.X=128 \
  model.hidden_mlp_dims.E=64 \
  model.hidden_mlp_dims.y=128 \
  model.hidden_dims.dx=256 \
  model.hidden_dims.de=64 \
  model.hidden_dims.dy=64 \
  model.hidden_dims.dim_ffX=256 \
  model.hidden_dims.dim_ffE=64 \
  model.hidden_dims.dim_ffy=256 \
  model.hidden_dims.n_head=8 \
  model.diffusion_steps=1000 \
  model.min_alpha=0.9999 \
  model.n_layers=10 \
  model.transition='uniform' \
  train.lr=0.0002 \
  train.n_epochs=1000 \
  train.batch_size=64 \
```

#### Sampling
After the experiment, you can run test (sampling) by executing `ddsbm-test`
This automatically load trained model's configuration.

```{bash}
DATASET_NAME=uncond_planar
EXP_NAME=2025-04-06_SB_planar_uniform
IDX=4
NUM_GPUS_TO_USE=2

ddsbm-test \
    dataset=$DATASET_NAME \
    general.test_only=./outputs/${DATASET_NAME}/${EXP_NAME}/forward_${IDX}/checkpoints/last.ckpt \
    general.gpus=${NUM_GPUS_TO_USE}
```

#### Analysis
The above code does the same sampling as community-20 and measures the metric at the same time.
