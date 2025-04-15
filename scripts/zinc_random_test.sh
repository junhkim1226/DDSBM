#!/bin/bash
#SBATCH -J DDSBM-TEST
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
NUM_GPUS_TO_USE=4
BATCH_SIZE=600

EXP_NAME=${EXP}_${MIN_ALPHA}

ddsbm-test \
    general.test_only=./outputs/zinc/2025-04-15_SB_0.999/forward_0/checkpoints/last.ckpt \
    general.gpus=${NUM_GPUS_TO_USE} \
