#############################################################################################################################
## RUN TESTS ON SIMULATION DATA #################################################################################################
## METHOD: multiCMH           #################################################################################################

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
    
## import 
from test_multiCMH import *

#############################################################################################################################
## simulation settings
import json
with open('tests/settings_Bbb.json', 'r') as f:
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
x_dsc = settings['x_dsc']
y_dsc = settings['y_dsc']
print(f"Output Directory: {output_dir}")
print(f"Number of Simulations: {nsim}")
print(f"NULL parameters: {null_p}, {null_N}, {null_n}, {null_P}")
print(f"ROC parameters: {roc_p}, {roc_N}, {roc_n}, {roc_P}")
os.makedirs(output_dir, exist_ok=True)

# methods to test
STTM_N = [5, 10, 15, 20, 25]
stt_method = 'medtree' # ward or medtree
METHODS = [f'{stt_method}_sttm_n{i}' for i in STTM_N]
# METHODS = [f"{method}_sttm_n{n}"
#            for method in ("medtree", "ward")
#            for n in STTM_N]
print(METHODS)



run_null = settings['run_null']
run_roc = settings['run_roc']
print(f'run_null {run_null}, run_roc {run_roc}')

#############################################################################################################################
## NULL SIMULATION
simtype = 'null'

if run_null:
    N = null_N  
    p = null_p
    for n_idx, n in enumerate(N):
        print(f'testing n = {n}, p = {p}')
        for method in METHODS:
            test_multiCMH(output_dir, simtype, n, p, nsim, method, alp, x_dsc, y_dsc)
                

    n = null_n 
    P = null_P
    for p_idx, p in enumerate(P):
        print(f'testing n = {n}, p = {p}')
        for method in METHODS:
            test_multiCMH(output_dir, simtype, n, p, nsim, method, alp, x_dsc, y_dsc)
                

#############################################################################################################################
## ROC SIMULATION
simtype = 'roc'

if run_roc:
    N = roc_N
    p = roc_p
    for n_idx, n in enumerate(N):
        print(f'testing n = {n}, p = {p}')
        for method in METHODS:
            test_multiCMH(output_dir, simtype, n, p, nsim, method, alp, x_dsc, y_dsc)
                

    n = roc_n
    P = roc_P
    for p_idx, p in enumerate(P):
        print(f'testing n = {n}, p = {p}')
        for method in METHODS:
            test_multiCMH(output_dir, simtype, n, p, nsim, method, alp, x_dsc, y_dsc)
                

