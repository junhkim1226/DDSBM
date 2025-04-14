#!/bin/bash

echo " python analysis/mmd_result.py ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_marginal_6/ ../outputs/SB/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_marginal_6/ results/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_marginal_6 --iterations {0..9..2} --dataset_name zinc "
python analysis/mmd_result.py \
    ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_marginal_6/ \
    ../outputs/SB/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_marginal_6/ \
    results/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_marginal_6 \
    --iterations {0..9..2} \
    --dataset_name zinc

echo " python analysis/mmd_result.py ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_uniform_6/ ../outputs/SB/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_uniform_6/ results/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_uniform_6 --iterations {0..9..2} --dataset_name zinc "
python analysis/mmd_result.py \
    ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_uniform_6/ \
    ../outputs/SB/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_uniform_6/ \
    results/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_uniform_6 \
    --iterations {0..9..2} \
    --dataset_name zinc

echo " python analysis/mmd_result.py ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_marginal_6/ ../outputs/SB/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_marginal_6/ results/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_marginal_6 --iterations {0..9..2} --dataset_name zinc "
python analysis/mmd_result.py \
    ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_marginal_6/ \
    ../outputs/SB/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_marginal_6/ \
    results/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_marginal_6 \
    --iterations {0..9..2} \
    --dataset_name zinc

echo " python analysis/mmd_result.py ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_uniform_6/ ../outputs/SB/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_uniform_6/ results/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_uniform_6 --iterations {0..9..2} --dataset_name zinc "
python analysis/mmd_result.py \
    ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_uniform_6/ \
    ../outputs/SB/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_uniform_6/ \
    results/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_uniform_6 \
    --iterations {0..9..2} \
    --dataset_name zinc

echo " python analysis/mmd_result.py ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_marginal_6/ ../outputs/SB/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_marginal_6/ results/ZINC250k_logp_2_4_random_matched_no_nH_0.99795_marginal_6 --iterations {0..9..2} --dataset_name zinc "
python analysis/mmd_result.py \
    ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_marginal_6/ \
    ../outputs/SB/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_marginal_6/ \
    results/ZINC250k_logp_2_4_random_matched_no_nH_0.99795_marginal_6 \
    --iterations {0..9..2} \
    --dataset_name zinc

echo " python analysis/mmd_result.py ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_uniform_6/ ../outputs/SB/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_uniform_6/ results/ZINC250k_logp_2_4_random_matched_no_nH_0.99795_uniform_6 --iterations {0..9..2} --dataset_name zinc "
python analysis/mmd_result.py \
    ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_uniform_6/ \
    ../outputs/SB/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_uniform_6/ \
    results/ZINC250k_logp_2_4_random_matched_no_nH_0.99795_uniform_6 \
    --iterations {0..9..2} \
    --dataset_name zinc

echo " python analysis/mmd_result.py ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_marginal_6/ ../outputs/SB/zinc/2024-09-07_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_marginal_6/ results/ZINC250k_logp_2_4_random_matched_no_nH_0.999_marginal_6 --iterations {0..9..2} --dataset_name zinc "
python analysis/mmd_result.py \
    ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_marginal_6/ \
    ../outputs/SB/zinc/2024-09-07_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_marginal_6/ \
    results/ZINC250k_logp_2_4_random_matched_no_nH_0.999_marginal_6 \
    --iterations {0..9..2} \
    --dataset_name zinc

echo " python analysis/mmd_result.py ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_uniform_6/ ../outputs/SB/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_uniform_6/ results/ZINC250k_logp_2_4_random_matched_no_nH_0.999_uniform_6 --iterations {0..9..2} --dataset_name zinc "
python analysis/mmd_result.py \
    ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_uniform_6/ \
    ../outputs/SB/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_uniform_6/ \
    results/ZINC250k_logp_2_4_random_matched_no_nH_0.999_uniform_6 \
    --iterations {0..9..2} \
    --dataset_name zinc
