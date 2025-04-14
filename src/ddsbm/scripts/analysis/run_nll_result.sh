#!/bin/bash

echo "python analysis/nll_result.py ../data/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_marginal_6/ ../data/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_marginal_6/ results/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_marginal_6/ --iterations {0..9..2}"
python analysis/nll_result.py \
    ../data/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_marginal_6/ \
    ../data/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_marginal_6/ \
    results/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_marginal_6/ \
    --iterations {0..9..2}

echo "python analysis/nll_result.py ../data/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_uniform_6/ ../data/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_uniform_6/ results/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_uniform_6/ --iterations {0..9..2}"
python analysis/nll_result.py \
    ../data/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_uniform_6/ \
    ../data/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_uniform_6/ \
    results/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.99795_uniform_6/ \
    --iterations {0..9..2}

echo "python analysis/nll_result.py ../data/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_marginal_6/ ../data/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_marginal_6/ results/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_marginal_6/ --iterations {0..9..2}"
python analysis/nll_result.py \
    ../data/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_marginal_6/ \
    ../data/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_marginal_6/ \
    results/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_marginal_6/ \
    --iterations {0..9..2}

echo "python analysis/nll_result.py ../data/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_uniform_6/ ../data/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_uniform_6/ results/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_uniform_6/ --iterations {0..9..2}"
python analysis/nll_result.py \
    ../data/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_uniform_6/ \
    ../data/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_uniform_6/ \
    results/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH_0.999_uniform_6/ \
    --iterations {0..9..2}

echo "python analysis/nll_result.py ../data/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_marginal_6/ ../data/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_marginal_6/ results/ZINC250k_logp_2_4_random_matched_no_nH_0.99795_marginal_6/ --iterations {0..9..2}"
python analysis/nll_result.py \
    ../data/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_marginal_6/ \
    ../data/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_marginal_6/ \
    results/ZINC250k_logp_2_4_random_matched_no_nH_0.99795_marginal_6/ \
    --iterations {0..9..2}

echo "python analysis/nll_result.py ../data/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_uniform_6/ ../data/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_uniform_6/ results/ZINC250k_logp_2_4_random_matched_no_nH_0.99795_uniform_6/ --iterations {0..9..2}"
python analysis/nll_result.py \
    ../data/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_uniform_6/ \
    ../data/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.99795_uniform_6/ \
    results/ZINC250k_logp_2_4_random_matched_no_nH_0.99795_uniform_6/ \
    --iterations {0..9..2}

echo "python analysis/nll_result.py ../data/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_marginal_6/ ../data/zinc/2024-09-07_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_marginal_6/ results/ZINC250k_logp_2_4_random_matched_no_nH_0.999_marginal_6/ --iterations {0..9..2}"
python analysis/nll_result.py \
    ../data/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_marginal_6/ \
    ../data/zinc/2024-09-07_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_marginal_6/ \
    results/ZINC250k_logp_2_4_random_matched_no_nH_0.999_marginal_6/ \
    --iterations {0..9..2}

echo "python analysis/nll_result.py ../data/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_uniform_6/ ../data/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_uniform_6/ results/ZINC250k_logp_2_4_random_matched_no_nH_0.999_uniform_6/ --iterations {0..9..2}"
python analysis/nll_result.py \
    ../data/zinc/2024-08-30_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_uniform_6/ \
    ../data/zinc/2024-09-06_zinc_ZINC250k_logp_2_4_random_matched_no_nH_0.999_uniform_6/ \
    results/ZINC250k_logp_2_4_random_matched_no_nH_0.999_uniform_6/ \
    --iterations {0..9..2}
