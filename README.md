# An Investigation of SLOE Estimation for Unbiased High-Dimensional Logistic Regression
#### Ameya Shere and Shuvadeep Saha

The purpose of this project is to test the SLOE estimator (unbiased high-dimensional logistic regression) for different kinds of data. For each type of data, we investigate the SLOE model's uncertainty (distribution of null p-values, prediction intervals) and evaluate its performance (F1 score, execution time), comparing against the Python statsmodels Logit model.

### Directory Structure
This codebase is structured as follows:
- `data` folder contains the UCI Heart Disease Dataset (Cleveland) from: <https://archive.ics.uci.edu/ml/datasets/heart+disease>
- `demo_plotting` folder contains scripts to generate plots for the Related Work section of the report
- `helpers` folder contains helper modules for running the simulations
    - `heart_data.py` contains a function to perform ETL on the UCI Heart Disease Dataset
    - `models_test.py` contains functions to fit the statsmodels Logit model and the SLOE model
    - `plotting.py` contains functions to generate p-value and prediction interval plots
    - `simulation.py` contains functions to generate the different kinds of data and run one simulation iteration
- `results` folder contains the results of the experiments
    - `plots` folder contains the p-value and prediction interval plots for the different kinds of data
    - `scores` folder contains a combined CSV file with alpha (logit inflation) values, F1 scores, and execution times for every combination of simulation parameters
    - `tables` folder contains refined CSV files from the scores CSV for use in generating results tables
- `sloe-logistic` folder contains the source code for the SLOE logistic regression estimator from: <https://github.com/google-research/sloe-logistic>
- `analysis.py` is the script that generates the results tables
- `run_experiments.py` is the script that runs all the experiments

```
.
├── data
├── demo_plotting
├── helpers
│   ├── heart_data.py
│   ├── models_test.py
│   ├── plotting.py
│   └── simulation.py
├── results
│   ├── plots
│   |   ├── gaussian
│   |   ├── heart
│   |   ├── heavy
│   |   ├── latent_gaussian
│   |   ├── latent_non_gaussian
│   |   └── non_gaussian
|   ├── tables
│   └── scores
├── sloe-logistic
├── analysis.py
└── run_experiments.py

```

### Instructions
1. `cd` to `./sloe-logistic`.
2. Build the module according to the instructions in the README in: <https://github.com/google-research/sloe-logistic>
3. `cd` to root.
4. To run experiments: `python run_experiments.py`.
5. Check `results` folder for experiment results.