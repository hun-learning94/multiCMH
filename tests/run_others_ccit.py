#############################################################################################################################
## RUN TESTS ON SIMULATION DATA #################################################################################################
## METHOD: CCIT          #################################################################################################

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
from test_others import *

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

run_null = settings['run_null']
run_roc = settings['run_roc']
print(f'run_null {run_null}, run_roc {run_roc}')

# methods to test
METHODS = ["CCIT"]
print(METHODS)

#############################################################################################################################
## NULL SIMULATION
if run_null:
    simtype = 'null'

    N = null_N
    p = null_p
    for n_idx, n in enumerate(N):
        print(f'testing n = {n}, p = {p}')
        for method in METHODS:
            test_others(output_dir, simtype, n, p, nsim, method, alp)

    n = null_n
    P = null_P
    for p_idx, p in enumerate(P):
        print(f'testing n = {n}, p = {p}')
        for method in METHODS:
            test_others(output_dir, simtype, n, p, nsim, method, alp)

#############################################################################################################################
## ROC SIMULATION
if run_roc:
    simtype = 'roc'

    N = roc_N
    p = roc_p   
    for n_idx, n in enumerate(N):
        print(f'testing n = {n}, p = {p}')
        for method in METHODS:
            test_others(output_dir, simtype, n, p, nsim, method, alp)

    n = roc_n
    P = roc_P
    for p_idx, p in enumerate(P):
        print(f'testing n = {n}, p = {p}')
        for method in METHODS:
            test_others(output_dir, simtype, n, p, nsim, method, alp)
