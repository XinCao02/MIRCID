"""
Example main.py workflow to demonstrate usage of the TIGER model in Python.
This mirrors the .Rmd you provided, including installation steps, data loading,
and running the model.

Ensure that:
  - 'tiger.py' is in the same directory (or installed as a module).
  - 'expr.rda' and 'prior.rda' contain the shapes (1780, 16) and (14, 1772), respectively.
"""

import os
import pandas as pd
import pyreadr  # conda install -c conda-forge pyreadr (to read R .rda)
# from cmdstanpy import install_cmdstan  # only if you need to install locally
from tiger import TIGER  # import the TIGER function we translated
import warnings
warnings.simplefilter("ignore", category=FutureWarning)


def main():
    # 1. (Optional) Install cmdstan if you do not already have it:
    # install_cmdstan()

    # 2. Load data from .rda
    # The .rda might contain multiple objects; check which ones are needed
    prior_rda = pyreadr.read_r('../data/CollecTRI_prior.rda')  # path to your prior.rda
    expr_rda = pyreadr.read_r('../data/trimmed_holland_rna_expr.rda')    # path to your expr.rda

    # Suppose the saved objects are exactly named "prior" and "expr"
    prior_df = list(prior_rda.values())[0]  # or prior_rda['prior'] if known
    expr_df = list(expr_rda.values())[0]    # or expr_rda['expr'] if known

    # In R code, prior has shape (14 x 1772), expr has shape (1780 x 16).
    # Usually after read, you'll get a DataFrame with the same row/col structure.

    # 3. Run TIGER with default parameters, for instance "method=MCMC", signed=False
    results = TIGER(
        TFexpressed=False,
        expr=expr_df,
        prior=prior_df,
        method="VB",    # "MCMC" or "VB"
        signed=True,    # True or False
        seed=42
    )

    # 4. Print or inspect results
    print("TFA score in the first 3 samples:")
    print(results["Z"].iloc[:, 0:3])  # Z is (TFs, samples), show first 3 samples

    print("W matrix (first 5 TFs x first 10 genes):")
    print(results["W"].iloc[0:10, 0:5])

    # Optionally save them to disk
    results["W"].to_csv("W_estimated.csv")
    results["Z"].to_csv("Z_estimated.csv")

if __name__ == "__main__":
    main()




### RUN MULTIPLE TIMES:


# import os
# import pandas as pd
# import numpy as np
# import pyreadr
# from tiger import TIGER  # Ensure TIGER is properly imported
# from tqdm import tqdm

# def load_data(prior_path, expr_path):
#     """
#     Load prior and expression data from .rda files.
#     """
#     prior_rda = pyreadr.read_r(prior_path)
#     expr_rda = pyreadr.read_r(expr_path)

#     prior_df = list(prior_rda.values())[0]
#     expr_df = list(expr_rda.values())[0]

#     return prior_df, expr_df

# def run_tiger(expr_df, prior_df, method="VB", signed=True, seed=42):
#     """
#     Run the TIGER model with given parameters and return W and Z matrices.
#     """
#     results = TIGER(
#         expr=expr_df,
#         prior=prior_df,
#         method=method,
#         signed=signed,
#         seed=seed
#     )
#     W = results["W"]
#     Z = results["Z"]
#     return W, Z

# def main():
#     # Paths to your data files
#     prior_path = '../data/prior.rda'
#     expr_path = '../data/expr.rda'

#     # Load data once
#     prior_df, expr_df = load_data(prior_path, expr_path)

#     # Determine the shape of W and Z for initializing arrays
#     sample_W, sample_Z = run_tiger(expr_df, prior_df, seed=42)
#     W_shape = sample_W.shape  # (genes, TFs)
#     Z_shape = sample_Z.shape  # (TFs, samples)

#     # Initialize arrays to store all W and Z matrices
#     all_W = np.zeros((W_shape[0], W_shape[1], 50))
#     all_Z = np.zeros((Z_shape[0], Z_shape[1], 50))

#     # Run TIGER 50 times with different seeds
#     pbar = tqdm(range(50))
#     for i in pbar:
#         seed = 42 + i  # Different seed for each run
#         print(f"Running TIGER iteration {i+1}/50 with seed {seed}...")
#         W, Z = run_tiger(expr_df, prior_df, seed=seed)
        
#         # Convert DataFrames to numpy arrays
#         all_W[:, :, i] = W.values
#         all_Z[:, :, i] = Z.values

#     # Calculate average and variance for W
#     avg_W = np.mean(all_W, axis=2)
#     var_W = np.var(all_W, axis=2)

#     # Calculate average and variance for Z
#     avg_Z = np.mean(all_Z, axis=2)
#     var_Z = np.var(all_Z, axis=2)

#     # Convert back to DataFrames for saving, preserving original indices and columns
#     avg_W_df = pd.DataFrame(avg_W, index=sample_W.index, columns=sample_W.columns)
#     var_W_df = pd.DataFrame(var_W, index=sample_W.index, columns=sample_W.columns)

#     avg_Z_df = pd.DataFrame(avg_Z, index=sample_Z.index, columns=sample_Z.columns)
#     var_Z_df = pd.DataFrame(var_Z, index=sample_Z.index, columns=sample_Z.columns)

#     # Save the results to CSV files
#     avg_W_df.to_csv("average_W.csv")
#     var_W_df.to_csv("variance_W.csv")
#     avg_Z_df.to_csv("average_Z.csv")
#     var_Z_df.to_csv("variance_Z.csv")

#     print("Completed 50 runs.")
#     print("Average and variance matrices have been saved as CSV files.")

# if __name__ == "__main__":
#     main()
