import torch
import numpy as np
import pandas as pd
from helpers.simulation import run_test, test_normalizing_flow
from helpers.heart_data import retrieve_heart_data
import warnings
warnings.filterwarnings('ignore')
np.random.seed(1)
torch.manual_seed(1)

# TEST NORMALIZING FLOW
print('NORMALIZING FLOW TEST')
test_normalizing_flow(3000, 600)

# SIMULATED DATA TESTS
print('SIMULATED DATA TESTS')
ratios = [0.1, 0.2]
n_vals = [500, 1000, 3000]
gamma = np.sqrt(5)
ci = 90

# To store F1 scores
scores = []

for ratio in ratios:
    for n in n_vals:
        is_max = False
        if ratio == max(ratios) and n == max(n_vals):
            is_max = True
        print('Trials for: n={}, ratio={}'.format(n, ratio))
        p = int(ratio*n)
        # TEST 1: Simulated Gaussian data
        print("TEST 1: Simulated Gaussian data")

        test_res = run_test(ratio, n, p, ci, 'Simulated Gaussian', 'gaussian', gamma=gamma, is_max=is_max)
        scores.append(test_res)

        # TEST 2: Simulated Non-Gaussian data
        print('TEST 2: Simulated Non-Gaussian data')

        test_res = run_test(ratio, n, p, ci, 'Simulated Non-Gaussian', 'non_gaussian', gamma=gamma, is_max=is_max)
        scores.append(test_res)

        # TEST 3: Latent Gaussian Variables
        print('TEST 3: Latent Gaussian Variables')

        test_res = run_test(ratio, n, p, ci, 'Simulated Latent Gaussian', 'latent_gaussian', gamma=gamma, is_max=is_max)
        scores.append(test_res)

        # TEST 4: Latent Non-Gaussian Variables
        print('TEST 4: Latent Non-Gaussian Variables')

        test_res = run_test(ratio, n, p, ci, 'Simulated Latent Non-Gaussian', 'latent_non_gaussian', gamma=gamma, is_max=is_max)
        scores.append(test_res)

        # TEST 5: Heavy tailed distribution
        print('TEST 5: Heavy Tailed Distribution')

        test_res = run_test(ratio, n, p, ci, 'Simulated Heavy-tailed distribution','heavy', gamma=gamma, is_max=is_max)
        scores.append(test_res)

# HEART DISEASE DATA
print('HEART DISEASE DATA TESTS')
sampling_ratio = 0.28

X, y = retrieve_heart_data(sampling_ratio)

n = X.shape[0]
p = X.shape[1]
ratio = np.around(p / float(n), decimals=2)
print('Trial for: n={}, ratio={}'.format(n, ratio))

test_res = run_test(ratio, n, p, ci, 'Heart Disease Data', 'heart', X=X, y=y)
scores.append(test_res)

# Write scores to file
scores_df = pd.DataFrame(scores)
print('Final Scores')
print(scores_df)
scores_df.to_csv('results/scores/scores.csv', index=False)