import os
root_marker = 'setup.py'
current_dir = os.getcwd()
if not os.path.exists(os.path.join(current_dir, root_marker)):
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    print(f"Working directory changed to: {os.getcwd()}")
else:
    print(f"Already at the root directory: {os.getcwd()}")

from multiCMH import multiCMH
import pandas as pd
import numpy as np
import os
import time
import math

def test_multiCMH(output_dir, simtype, n, p, nsim, method, alp, x_dsc = False, y_dsc = False):

    prefix = f'{simtype}_n{n}p{p}'
    pvals_filename_full = os.path.join(output_dir, prefix + f'_{method}_pvals.csv')
    times_filename_full = os.path.join(output_dir, prefix + f'_{method}_times.csv')

    Times = []
    Pvals = []

    for sim in range(nsim):
        # Define filenames for the current simulation
        pvals_filename_sim = os.path.join(output_dir, prefix + f'_{method}_pvals_sim{sim}.csv')
        times_filename_sim = os.path.join(output_dir, prefix + f'_{method}_times_sim{sim}.csv')

        # Check if results for this simulation already exist
        # if os.path.exists(pvals_filename_sim) and os.path.exists(times_filename_sim) and "time" in output_dir:
        #     try:
        #         # Load the existing results to maintain the cumulative list
        #         pval = pd.read_csv(pvals_filename_sim, header=None).iloc[0, 0]
        #         time_val = pd.read_csv(times_filename_sim, header=None).iloc[0, 0]
        #         Pvals.append(pval)
        #         Times.append(time_val)
        #         print(f"Results for sim {sim} with method {method} already exist.")
        #         continue # Skip to the next simulation
        #     except Exception as e:
        #         pass

        if sim % 10 == 0:
            print(f"{method} sim = {sim}", end = ' ')

        # Read in data
        filename = os.path.join(output_dir, prefix + f'_sim{sim}.csv')
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            Pvals.append(np.nan)
            Times.append(np.nan)
            continue

        x = df['x'].to_numpy()
        y = df['y'].to_numpy()
        Z = df.drop(columns=['x', 'y']).values

        try:
            tic = time.process_time()
            
            stt_method, _, sttm_n = method.partition('_sttm_n')
            sttm_n = int(sttm_n)
            scr_all, scr_mrg = 20, 10
            k1 = k2 = min(math.floor(math.log2(n / scr_mrg)), 7)
            # print(f'k1 {k1} k2 {k2}', end = ' ')
            res = multiCMH(x=x, 
                           y=y, 
                           Z=Z,
                           x_dsc=x_dsc, 
                           y_dsc=y_dsc, 
                           k1=k1, 
                           k2=k2, 
                           alp=alp,
                           sttm_n=sttm_n, 
                           stt_method=stt_method,
                           scr_all=scr_all, 
                           scr_mrg=scr_mrg, 
                           verbose=False)
                
            pval = res.p_value
            
            toc = time.process_time()
            
            if 'time' in output_dir:
                print(f'sim {sim} elapsed {toc-tic}')
                
            Pvals.append(pval)
            Times.append(toc - tic)

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