#!/bin/bash
#SBATCH -J DDSBM-SB-0.999
#SBATCH -p a4000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=1000:00:00
#SBATCH --gres=gpu:4

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ddsbm

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
export NCCL_P2P_DISABLE=1

EXP=SB
DATASET=zinc
MIN_ALPHA=0.999
EPOCHS=300

EXP_NAME=${EXP}_${MIN_ALPHA}

ddsbm-train \
    dataset.name=${DATASET} \
    general.name=${EXP_NAME} \
    general.gpus=${NUM_GPUS_TO_USE} \
    model.min_alpha=${MIN_ALPHA} \
    train.n_epochs=${EPOCHS}
