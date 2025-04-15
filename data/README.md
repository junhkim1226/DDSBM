# ZINC dataset preprocessing

We have processed raw ZINC250k data by removing non-neutral molecules and adding some properties, which is `raw/zinc250k_ions_removed_properties.csv`.
To make the zinc datasets we used in the main text, (random-matched and Tanimoto similarity-based matched), one can run `make_logp_data.ipynb` and then `couople_zinc250k.py`.

1. `make_logp_data.ipynb`

This notebook processes the `raw/zinc250k_ions_removed_properties.csv` file, which contains molecules with pre-computed molecular properties (e.g., logP, QED, etc.) and excludes non-neutral molecules (ions removed).

Main functionalities:
- Filters the dataset with other criteria to get a dataset without charged molecules or molecules with `nH`.
- Splits the dataset into two groups based on the distribution of logP values (e.g., low vs. high logP).
- Generates, `raw/ZINC250k_logp_2_4_random_matched_no_nH.csv`, which are randomly matched pairs of zinc molecules with logP distributions around 2 and 4.

2. `couple_zinc250k.py`

This script performs molecule pairing using the Hungarian algorithm guided by molecular similarity. (Appendix F.2.2)

Main functionalities:
- Loads molecular fingerprints (e.g., Morgan fingerprints) from the processed dataset.
- Computes pairwise Tanimoto similarity scores between molecules in the two logP-divided sets.
- Applies the Hungarian matching algorithm to maximize similarity-based pairing.
- Generates `raw/ZINC250k_logp_2_4_tanimoto_sim_matched_no_nH.csv`.
