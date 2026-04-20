import os
root_marker = 'setup.py'
current_dir = os.getcwd()
if not os.path.exists(os.path.join(current_dir, root_marker)):
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    print(f"Working directory changed to: {os.getcwd()}")
else:
    print(f"Already at the root directory: {os.getcwd()}")

# CCIT (Sen 2017) pip install CCIT==0.4
from CCIT import CCIT
# LPCIT (Scetbon 2022)
from LPCIT import lp_ci_test
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.data_processing import DataFrame

import pandas as pd
import numpy as np
import time
import sys
import warnings

# Define a function to suppress output
def suppress_stdout_warnings(func, *args, **kwargs):
    """A context manager to suppress standard output and warnings."""
    # Open a null file to redirect stdout
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    sys.stdout = devnull
    # Temporarily suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Call the original function
            result = func(*args, **kwargs)
        finally:
            # Restore stdout and close the null file
            sys.stdout = old_stdout
            devnull.close()
    return result

def test_others(output_dir, simtype, n, p, nsim, method, alp=0.05):
    """
    Args:
        simtype (str): The type of simulation being run (e.g., 'normal', 'poisson').
        n (int): The number of samples.
        p (int): The number of features.
        nsim (int): The number of simulations to run.
        method (str): The name of others to test.
        output_dir (str): The directory where the simulation data and results are stored.
        alp (float): The significance level for the statistical test.
    """
    prefix = f'{simtype}_n{n}p{p}'
    pvals_filename_full = os.path.join(output_dir, prefix + f'_{method}_pvals.csv')
    times_filename_full = os.path.join(output_dir, prefix + f'_{method}_times.csv')

    # Initialize lists to store results for this method
    Pvals = []
    Times = []

    for sim in range(nsim):
        # Define filenames for the current simulation
        pvals_filename_sim = os.path.join(output_dir, prefix + f'_{method}_pvals_sim{sim}.csv')
        times_filename_sim = os.path.join(output_dir, prefix + f'_{method}_times_sim{sim}.csv')

        # Check if results for this simulation already exist
        if os.path.exists(pvals_filename_sim) and os.path.exists(times_filename_sim):
            try:
                # Load the existing results to maintain the cumulative list
                pval = pd.read_csv(pvals_filename_sim, header=None).iloc[0, 0]
                time_val = pd.read_csv(times_filename_sim, header=None).iloc[0, 0]
                Pvals.append(pval)
                Times.append(time_val)
                print(f"Results for sim {sim} with method {method} already exist. Skipping this.")
                continue  # Skip to the next simulation
            except Exception as e:
                pass

        if sim % 10 == 0:
            print(f"{method} sim = {sim}", end = ' ')

        # Read in data
        filename = os.path.join(output_dir, prefix + f'_sim{sim}.csv')
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            print(f"File not found: {filename}. Skipping simulation {sim}.")
            # Append NaN for this simulation to keep the lists aligned
            Pvals.append(np.nan)
            Times.append(np.nan)
            continue

        x = df['x'].to_numpy()
        y = df['y'].to_numpy()
        Z = df.drop(columns=['x', 'y']).values

        try:
            tic = time.process_time()

            # other methods
            if method == "CCIT":
                pval = CCIT.CCIT(x.reshape(-1, 1), y.reshape(-1, 1), Z, num_iter=30, bootstrap=True, nthread=20)
            elif method == "LPCIT":
                # res = lp_ci_test.test_asymptotic_ci(x.reshape(-1, 1), y.reshape(-1, 1), Z, rank=1000)
                # pval = float(res['pvalue'])
                res = suppress_stdout_warnings(
                    lp_ci_test.test_asymptotic_ci,
                    x.reshape(-1, 1), y.reshape(-1, 1), Z, rank=1000)
                pval = float(res['pvalue'])
            elif method == "CMIknn":
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)
                data = np.hstack([x, y, Z])
                cmi_test = CMIknn()
                dataframe = DataFrame(data=data)
                cmi_test.set_dataframe(dataframe)
                X_cmi = [(0, 0)]  # Column 0 (x)
                Y_cmi = [(1, 0)]  # Column 1 (y)
                Z_cmi = [(i, 0) for i in range(2, 2 + Z.shape[1])]  # The columns for z
                _, pval = cmi_test.run_test(X=X_cmi, Y=Y_cmi, Z=Z_cmi)
            
            toc = time.process_time()
            Pvals.append(pval)
            Times.append(toc - tic)
            if 'time' in output_dir:
                print(f'sim {sim} elapsed {toc-tic}')

        except Exception as e:
            print(f"{method} failed for sim {sim}: {e}. p-value and time set to NaN.")
            Pvals.append(np.nan)
            Times.append(np.nan)
            pass

        # Save results for each simulation
        pd.Series([Pvals[-1]]).to_csv(pvals_filename_sim, index=False, header=False)
        pd.Series([Times[-1]]).to_csv(times_filename_sim, index=False, header=False)

        # Checkpoint: save cumulative results every 10th simulation (9, 19, 29, ...)
        if sim % 10 == 9:
            print(f"Saving cumulative results up to sim {sim}...")
            pd.Series(Pvals).to_csv(pvals_filename_full, index=False, header=False)
            pd.Series(Times).to_csv(times_filename_full, index=False, header=False)

    # Final save of all results after the loop completes
    pd.Series(Pvals).to_csv(pvals_filename_full, index=False, header=False)
    pd.Series(Times).to_csv(times_filename_full, index=False, header=False)