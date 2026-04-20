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
with open('tests/settings.json', 'r') as f:
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

# the higher marginal correlation is, the harder it is for methods to correctly reject the null,
# and our multiCMH method is especially vulnerable to this setting
g_pool = [
    lambda x: x,
    lambda x: x**2,
    lambda x: x**3,
    np.tanh,
    lambda x: np.exp(-np.abs(x))
]

def gen_null(n, p, seed=None):
    rng = np.random.default_rng(seed)
    f1_list = rng.choice(g_pool, size=n)
    f2_list = rng.choice(g_pool, size=n)
    Z = rng.standard_normal((n, p))
    Z_bar = Z[:, :(p//2)].mean(axis=1)  # shape (n,)
    X_input = Z_bar + rng.standard_normal(n)
    X = np.array([f1_list[i](X_input[i]) for i in range(n)])
    Y_input = Z_bar + rng.standard_normal(n)
    Y = np.array([f2_list[i](Y_input[i]) for i in range(n)])
    return X, Y, Z

def gen_alt(n, p, seed=None):
    rng = np.random.default_rng(seed)
    f1_list = rng.choice(g_pool, size=n)
    f2_list = rng.choice(g_pool, size=n)
    Z = rng.standard_normal((n, p))
    eps_b = rng.standard_normal(n) * 0.8
    X_input = eps_b + rng.standard_normal(n)
    X = np.array([f1_list[i](X_input[i]) for i in range(n)])
    Y_input = eps_b + rng.standard_normal(n)
    Y = np.array([f2_list[i](Y_input[i]) for i in range(n)])
    return X, Y, Z

def is_alt(sim, nsim):
    return (sim+1) > math.floor(nsim / 2) # true is alternative, false is null

#############################################################################################################################
## NULL SIMULATION
N = null_N
p = null_p
for n_idx, n in enumerate(N):
    print(f'sampling n = {n}, p = {p}')
    for sim in range(nsim):
        x, y, Z = gen_null(n, p, seed = sim + null_seed)
        df = pd.DataFrame({
            'x':x,
            'y':y
        })
        Z_df = pd.DataFrame(Z, columns = [f'Z{i+1}' for i in range(p)])
        df = pd.concat([df, Z_df], axis = 1)
        # save
        filename = f'null_n{n}p{p}_sim{sim}.csv'
        filename = os.path.join(output_dir, filename)
        df.to_csv(filename, index=False)
        
n = null_n
P = null_P
for p_idx, p in enumerate(P):
    print(f'sampling n = {n}, p = {p}')
    for sim in range(nsim):
        x, y, Z = gen_null(n, p, seed = sim + null_seed)
        df = pd.DataFrame({
            'x':x,
            'y':y
        })
        Z_df = pd.DataFrame(Z, columns = [f'Z{i+1}' for i in range(p)])
        df = pd.concat([df, Z_df], axis = 1)
        # save
        filename = f'null_n{n}p{p}_sim{sim}.csv'
        filename = os.path.join(output_dir, filename)
        df.to_csv(filename, index=False)

#############################################################################################################################
## ROC SIMULATION
N = roc_N
p = roc_p
for n_idx, n in enumerate(N):
    print(f'sampling n = {n}, p = {p}')
    for sim in range(nsim):
        if is_alt(sim, nsim):
            # simulate from alternative
            x, y, Z = gen_alt(n, p, seed = sim + roc_seed)
        else:
            # simulate from null
            x, y, Z = gen_null(n, p, seed = sim + roc_seed)
        # save
        df = pd.DataFrame({
            'x':x,
            'y':y
        })
        Z_df = pd.DataFrame(Z, columns = [f'Z{i+1}' for i in range(p)])
        df = pd.concat([df, Z_df], axis = 1)
        # save
        filename = f'roc_n{n}p{p}_sim{sim}.csv'
        filename = os.path.join(output_dir, filename)
        df.to_csv(filename, index=False)
        
n = roc_n
P = roc_P
for p_idx, p in enumerate(P):
    print(f'sampling n = {n}, p = {p}')
    for sim in range(nsim):
        if is_alt(sim, nsim):
            # simulate from alternative
            x, y, Z = gen_alt(n, p, seed = sim + roc_seed)
        else:
            # simulate from null
            x, y, Z = gen_null(n, p, seed = sim + roc_seed)
        # save
        df = pd.DataFrame({
            'x':x,
            'y':y
        })
        Z_df = pd.DataFrame(Z, columns = [f'Z{i+1}' for i in range(p)])
        df = pd.concat([df, Z_df], axis = 1)
        # save
        filename = f'roc_n{n}p{p}_sim{sim}.csv'
        filename = os.path.join(output_dir, filename)
        df.to_csv(filename, index=False)