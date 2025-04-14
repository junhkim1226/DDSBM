#!/bin/bash

echo "python analysis/only_sb_prop_result.py ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_marginal_6/  results/train_only_sb_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_marginal_6 --iterations {0..3}"
python analysis/only_sb_prop_result.py \
    ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_marginal_6/ \
    results/train_only_sb_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_marginal_6 \
    --iterations {0..3}

echo "python analysis/only_sb_prop_result.py ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_uniform_6/  results/train_only_sb_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_uniform_6 --iterations {0..3}"
python analysis/only_sb_prop_result.py \
    ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_uniform_6/ \
    results/train_only_sb_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_uniform_6 \
    --iterations {0..3}

echo "python analysis/only_sb_prop_result.py ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_marginal_6/  results/train_only_sb_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_marginal_6 --iterations {0..3}"
python analysis/only_sb_prop_result.py \
    ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_marginal_6/ \
    results/train_only_sb_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_marginal_6 \
    --iterations {0..3}

echo "python analysis/only_sb_prop_result.py ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_uniform_6/  results/train_only_sb_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_uniform_6 --iterations {0..3}"
python analysis/only_sb_prop_result.py \
    ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_uniform_6/ \
    results/train_only_sb_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_uniform_6 \
    --iterations {0..3}

echo "python analysis/only_sb_prop_result.py ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_marginal_6/  results/train_only_sb_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_marginal_6 --iterations {0..3}"
python analysis/only_sb_prop_result.py \
    ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_marginal_6/ \
    results/train_only_sb_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_marginal_6 \
    --iterations {0..3}

echo "python analysis/only_sb_prop_result.py ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_uniform_6/  results/train_only_sb_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_uniform_6 --iterations {0..3}"
python analysis/only_sb_prop_result.py \
    ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_uniform_6/ \
    results/train_only_sb_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_uniform_6 \
    --iterations {0..3}

echo "python analysis/only_sb_prop_result.py ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_marginal_6/  results/train_only_sb_ZINC250k_logp_2_4_random_matched_no_nH_0.999_marginal_6 --iterations {0..3}"
python analysis/only_sb_prop_result.py \
    ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_marginal_6/ \
    results/train_only_sb_ZINC250k_logp_2_4_random_matched_no_nH_0.999_marginal_6 \
    --iterations {0..3}

echo "python analysis/only_sb_prop_result.py ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_uniform_6/  results/train_only_sb_ZINC250k_logp_2_4_random_matched_no_nH_0.999_uniform_6 --iterations {0..3}"
python analysis/only_sb_prop_result.py \
    ../outputs/SB/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_uniform_6/ \
    results/train_only_sb_ZINC250k_logp_2_4_random_matched_no_nH_0.999_uniform_6 \
    --iterations {0..3}
