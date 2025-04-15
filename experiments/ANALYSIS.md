# ANALYSIS
> [!NOTE]
> `PROJECT_DIR` is [here](..). Also, the analysis codes assumes that you did **not** modified any codes in `../src/main.py`.

## Getting Mol Object from tensor of graphs
After sampling, you will get `.pt` file with generated molecule graphs in **`${YOUR_RESULT_PATH}`**, which is `${PROJECT_DIR}/outputs/${DATASET_NAME}/${EXP_NAME}/test_${direction}_${iteration}_last/generated_joint_test_seed${SEED}.pt`. To analyze the result, we should convert this graph data into RDKit Mol object. You can do this by `./graph_to_mol.py`.

```{bash}
# SB experiment
iteration=  # SB outer loop iteration
epoch=last  # SB always use the last epoch
SEED=42
file=${PROJECT_DIR}/outputs/${DATASET_NAME}/${EXP_NAME}/test_${direction}_${iteration}_${epoch}/generated_joint_test_seed${SEED}.pt

# Bridge experiment
iteration=0  # Bridge are trained in 0'th iteration only
epoch=  # same iteration as SB ((SB_iteration * epochs) - 1)
SEED=42  # generation seed
file=${PROJECT_DIR}/outputs/${DATASET_NAME}/${EXP_NAME}/test_${direction}_${iteration}_${epoch}/generated_joint_test_seed${SEED}.pt

direction=  # forward or backward

python graph_to_mol.py \
    $file \
    --num_workers 1  # TO do multiprocessing set this value as number of processes
```

This will generate resulting molecule smiles in `${PROJECT_DIR}/outputs/${DATASET_NAME}/${EXP_NAME}/test_${direction}_${iteration}_${epoch}/result_${direction}_seed${SEED}.csv`. We also note that the resulting validity, uniqueness, novelty is written in `${PROJECT_DIR}/outputs/${DATASET_NAME}/${EXP_NAME}/test_${direction}_${iteration}_${epoch}/VUN_${direction}_${SEED}.txt`.

> [!IMPORTANT]
> By default, the analysis results will be generated in `${PROJECT_DIR}/results/{nll,fcd,nspdk,prop_diff,val_props}`. One can change this output directory with `--output_path` option, for all the codes introduced in below.

## Negative Log Likelihood (NLL) Analysis
Since we always generate `pandas.DataFrame` for NLL in `src/ddsbm/graph_match_helper.py`, we can gather pre-computed ones and make result from them.
`${EXP_NAME}` is experiment name, which defined in [Training Section](../README.md#Training), which automatically generated via running `ddsbm-train (src/main.py)`.
You can set specific iterations to analyze with `--iterations` flag.

```{bash}
ITERATIONS=$(seq 0 5)  # 0 1 2 3 4 5
DIRECTION=forward

python analysis/nll.py \
    --data_path ${PROJECT_DIR}/data/${DATASET_NAME}/${EXP_NAME} \
    --seeds ${SEED} \
    --direction ${DIRECTION} \
    --iterations ${ITERATIONS}
```

This will make `${PROJECT_DIR}/results/nll/${DATASET_NAME}-${EXP_NAME}-${DIRECTION}-${SEEDS}.csv`.
```bash
../results/
└── nll
    └── zinc-2025-04-15_SB_0.999-forward-42.csv
```

## Neighborhood Subgraph Pairwise Distance Kernel (NSPDK) Analysis
`analysis/nspdk.py` compare NSPDK between original and generated data. You can set specific iterations to analyze with `--iterations` flag. For reproducibility, you should fix `PYTHONHASHSEED` each time running the evaluation code.

```{bash}
ITERATIONS=$(seq 0 5)  # 0 1 2 3 4 5
DIRECTION=forward

PYTHONHASHSEED=0 python analysis/nspdk.py \
    --experiment_path ${PROJECT_DIR}/outputs/${DATASET_NAME}/${EXP_NAME} \
    --seeds ${SEED} \
    --direction ${DIRECTION} \
    --iterations ${ITERATIONS} \
    --num_workers 1  # Number of multiprocessing workers (>1 does not work for now)
```

This will make `${PROJECT_DIR}/results/nspdk/${DATASET_NAME}-${EXP_NAME}-${DIRECTION}-${SEEDS}.csv`.
```bash
../results/
└── nspdk
    └── zinc-2025-04-15_SB_0.999-forward-42.csv
```

## Property and Validity Analysis
For property evaluation, we need to use two codes: `analysis/val_prop_wd.py` and `analysis/prop_diff.py`.

`analysis/val_prop_wd.py` compare Wasserstein distance between original data's property distribution and generated data's property distribution with validity of generations. You can set specific iterations to analyze with `--iterations` flag.

```{bash}
ITERATIONS=$(seq 0 5)  # 0 1 2 3 4 5
DIRECTION=forward

python analysis/val_prop_wd.py \
    --experiment_path ${PROJECT_DIR}/outputs/${DATASET_NAME}/${EXP_NAME} \
    --seeds ${SEED} \
    --direction ${DIRECTION} \
    --iterations ${ITERATIONS} \
    --num_workers 4  # Number of multiprocessing workers (No need to set high value for logp calculation)
```

This will make `${PROJECT_DIR}/results/val_props/${DATASET_NAME}-${EXP_NAME}-${DIRECTION}-${SEEDS}.csv`.

`analysis/prop_diff.py` compare Mean Absolute Deviation (MAD) between original data's property distribution and generated data's property distribution with validity of generations. You can set specific iterations to analyze with `--iterations` flag.

```{bash}
ITERATIONS=$(seq 0 5)  # 0 1 2 3 4 5
DIRECTION=forward

python analysis/prop_diff.py \
    --experiment_path ${PROJECT_DIR}/outputs/${DATASET_NAME}/${EXP_NAME} \
    --seeds ${SEED} \
    --direction ${DIRECTION} \
    --iterations ${ITERATIONS} \
    --num_workers 4  # Number of multiprocessing workers (No need to set high value for logp calculation)
```

This will make `${PROJECT_DIR}/results/prop_diff/${DATASET_NAME}-${EXP_NAME}-${DIRECTION}-${SEEDS}.csv`.

```bash
../results/
└── val_props
    ├── zinc-2025-04-15_SB_0.999-forward-val-42.csv
    └── zinc-2025-04-15_SB_0.999-forward-wd-42.csv
```

## Fréchet ChemNet Distance (FCD) Analysis
FCD Analysis do very similar thing compared to [Property and Validity Analysis](#property-and-validity-analysis), thus inputs are the same. You can set specific iterations to analyze with `--iterations` flag.

>[!NOTE]
> We highly recommend to use GPU for computing fcd.

```{bash}
ITERATIONS=$(seq 0 5)  # 0 1 2 3 4 5
DIRECTION=forward

python analysis/fcd.py \
    --experiment_path ${PROJECT_DIR}/outputs/${DATASET_NAME}/${EXP_NAME} \
    --seeds ${SEED} \
    --direction ${DIRECTION} \
    --iterations ${ITERATIONS} \
    --num_workers 4
```

This will make `${PROJECT_DIR}/results/fcd/${DATASET_NAME}-${EXP_NAME}-${DIRECTION}-${SEEDS}.csv`.

```bash
../results/
└── fcd
    └── zinc-2025-04-15_SB_0.999-forward-42.csv
```
