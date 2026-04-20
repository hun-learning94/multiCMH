## set working directory to the root
import os
root_marker = 'setup.py'
current_dir = os.getcwd()
if not os.path.exists(os.path.join(current_dir, root_marker)):
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    print(f"Working directory changed to: {os.getcwd()}")
else:
    print(f"Already at the root directory: {os.getcwd()}")
    
## import stuff
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#############################################################################################################################
## simulation settings
import json
with open('tests/settings_time2.json', 'r') as f:
    settings = json.load(f)
output_dir = settings['output_dir']
alp = settings['alp']
nsim = settings['nsim']
null_seed = settings['null_seed']
null_p = settings['null_p']
null_N = settings['null_N']
null_n = settings['null_n']
null_P = settings['null_P']
roc_seed = settings['roc_seed']
roc_p = null_p
roc_N = null_N
roc_n = null_n
roc_P = null_P
print(f"Output Directory: {output_dir}")
print(f"Number of Simulations: {nsim}")
print(f"NULL parameters: {null_p}, {null_N}, {null_n}, {null_P}")
print(f"ROC parameters: {roc_p}, {roc_N}, {roc_n}, {roc_P}")
os.makedirs(output_dir, exist_ok=True)

def gen_null(n, p, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    Y = rng.standard_normal(n)
    Z = rng.standard_normal((n,p))
    return X, Y, Z


#############################################################################################################################
## NULL SIMULATION
N = null_N
p = null_p
for n_idx, n in enumerate(N):
    print(f'sampling n = {n}, p = {p}')
    for sim in range(nsim):
        filename = f'null_n{n}p{p}_sim{sim}.csv'
        filename = os.path.join(output_dir, filename)
        if os.path.exists(filename):
            print(f"Skipping existing file: {filename}")
            continue
        x, y, Z = gen_null(n, p, seed = sim + null_seed)
        df = pd.DataFrame({
            'x':x,
            'y':y
        })
        Z_df = pd.DataFrame(Z, columns = [f'Z{i+1}' for i in range(p)])
        df = pd.concat([df, Z_df], axis = 1)
        # save
        df.to_csv(filename, index=False)
        
