import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

def ks_statistic(p_values):
    # Remove NaN values from the input p_values
    p_values_filtered = np.asarray(p_values)
    p_values_filtered = p_values_filtered[~np.isnan(p_values_filtered)]
    nsim = len(p_values_filtered)
    # If there are no valid p-values, return 0
    if nsim == 0:
        return 0
    p_values_filtered = np.sort(p_values_filtered)
    empirical_cdf = np.arange(1, nsim + 1) / nsim
    ks = np.max(np.abs(empirical_cdf - p_values_filtered))
    return ks

def is_alt(sim, nsim):
    return (sim+1) > math.floor(nsim / 2) # true is alternative, false is null

def compute_roc(oracle, pvals):
    # remove NaNs
    combined = np.column_stack((oracle, pvals))
    valid_rows = ~np.isnan(combined).any(axis=1)
    oracle = combined[valid_rows, 0].astype(bool)
    pvals = combined[valid_rows, 1]
    # If there are no valid p-values after filtering, return 0 for AUC
    if len(pvals) == 0:
        return [], [], 0
    alps = np.linspace(0, 1, num=200)
    pvals_null = pvals[~oracle]
    pvals_alt = pvals[oracle]
    FPR = [np.mean(pvals_null <= alp) for alp in alps]
    TPR = [np.mean(pvals_alt <= alp) for alp in alps]
    return FPR, TPR, np.trapezoid(TPR, FPR)

def plot_rocs(ax, output_dir, n, p, METHODS, nsim, color_map, oracle, alpha=0.05, show_legend=False):
    prefix = 'roc'
    method_name_map = {
        'medtree_sttm_n15':'multiCMH',
        'medtree_sttm_n12':'multiCMH',
        'medtree_sttm_n10':'multiCMH',
        'medtree_sttm_n5':'multiCMH'
    }
    # Check if there is any data to plot
    has_data = False
    linestyles = ['--', ':', '-.', (0, (3, 1,1,1,1,1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (5, (10,3))]
    
    # Process data and plot
    for i, method in enumerate(METHODS):
        linestyle = linestyles[i % len(linestyles)]
        if 'medtree' in method:
            linestyle = '-'
        filename = os.path.join(output_dir, f"{prefix}_n{n}p{p}_{method}_pvals.csv")
        try:
            pvals = np.genfromtxt(
                filename,
                delimiter=",",
                dtype=float,
                filling_values=np.nan,  # replaces missing entries with NaN
                skip_header=0,          # adjust if there's a header
                invalid_raise=False     # prevents crash on bad rows
                )

            if len(pvals) < nsim:
                pvals = np.concatenate([pvals, np.full(nsim - len(pvals), np.nan)])
            # Filter out NaNs and check if there's any data left
            if pvals.size > 0 and not np.all(np.isnan(pvals)):
                FPR, TPR, _ = compute_roc(oracle, pvals)
                ax.plot(FPR, TPR, label= method_name_map.get(method, method), color = color_map.get(method, 'gray'),
                        linewidth = 1.25, linestyle = linestyle)
                has_data = True
        except (IOError, ValueError) as e:
            # File not found or corrupt, plotmat value remains nan
            print(f"Error doing file '{filename}': {e}")
            pass
            
    # If no valid data was found for any method, turn off the axis
    if not has_data:
        ax.axis('off')
        return

    # Setup plot if there is data
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.02, 1.02)
    x = np.linspace(0, 1, 100)
    ax.plot(x, x, color='gray', linestyle='dotted', linewidth=1.25)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(f'n={n}, d={p}')
    if show_legend:
        ax.legend(ncol=2, fontsize='small', columnspacing=0.8)
    
    
def plot_ecdf(ax, output_dir, n, p, METHODS, nsim, color_map):
    prefix = 'null'
    has_data = False
    method_name_map = {
        'medtree_sttm_n15':'multiCMH',
        'medtree_sttm_n12':'multiCMH',
        'medtree_sttm_n10':'multiCMH',
        'medtree_sttm_n5':'multiCMH'
    }
    linestyles = ['--', ':', '-.', (0, (3, 1,1,1,1,1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (5, (10,3))]
    
    for i, method in enumerate(METHODS):
        linestyle = linestyles[i % len(linestyles)]
        if 'medtree' in method:
            linestyle = '-'
        filename = os.path.join(output_dir, f"{prefix}_n{n}p{p}_{method}_pvals.csv")
        try:
            pvals = np.genfromtxt(
                filename,
                delimiter=",",
                dtype=float,
                filling_values=np.nan,  # replaces missing entries with NaN
                skip_header=0,          # adjust if there's a header
                invalid_raise=False     # prevents crash on bad rows
                )

            if len(pvals) < nsim:
                pvals = np.concatenate([pvals, np.full(nsim - len(pvals), np.nan)])
            pvals = pvals[~np.isnan(pvals)]
            if pvals.size > 0:
                has_data = True
                sorted_pvals = np.sort(pvals)
                ecdf = np.arange(1, len(sorted_pvals) + 1) / len(sorted_pvals)
                ax.plot(sorted_pvals, ecdf,
                        label = method_name_map.get(method, method),
                        linestyle=linestyle,
                        color=color_map.get(method, 'gray'),
                        linewidth=1.25)
        except (IOError, ValueError) as e:
            print(f"Error reading file '{filename}': {e}")
            continue
    if not has_data:
        ax.axis('off')
        return
    x = np.linspace(0, 1, 100)
    ax.plot(x, x, color='black', linestyle='dotted', linewidth=1.25)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('p-value')
    ax.set_ylabel('ECDF of p-value')
    ax.set_title(f'n={n}, d={p}')
    # ax.legend()
    

# nsim = 100
# oracle = is_alt(np.arange(0, nsim), nsim)
# methods = [f'method{i}' for i in [1,2,3]]
# pvals = [np.random.uniform(0, 1, nsim) for _ in methods]
# dict_pvals = dict(zip(methods, pvals))
# plot_rocs(10, 2, oracle, dict_pvals)

def plot_lines(ax, output_dir, prefix, N, P, ylab, METHODS, nsim ,color_map, alpha=0.05, oracle=None, show_legend = True):
    
    method_name_map = {
        'medtree_sttm_n15':'multiCMH',
        'medtree_sttm_n12':'multiCMH',
        'medtree_sttm_n10':'multiCMH',
        'medtree_sttm_n5':'multiCMH'
    }
    
    # method_name_map = {
    #     'medtree_sttm_n25':'eta25',
    #     'medtree_sttm_n20':'eta20',
    #     'medtree_sttm_n15':'eta15',
    #     'medtree_sttm_n10':'eta10',
    #     'medtree_sttm_n5':'eta5'
    # }
    
    """
    output_dir: Directory for output files
    prefix: 'null' or 'roc'
    N, P: Either (N, p) or (n, P)
    ylab: 'Probability of rejection', 'KS statistic', or 'AUROC'
    METHODS: List of method names
    """
    # Input validations
    if prefix not in {'null', 'roc'}:
        raise ValueError("prefix must be 'null' or 'roc'")
    if ylab not in {'Probability of rejection', 'KS statistic', 'AUROC', 'Median log CPU time'}:
        raise ValueError("invalid ylab")
    if ylab == 'AUROC' and oracle is None:
        raise ValueError('AUROC needs oracle')

    # Determine fixed parameter type
    if isinstance(N, list) and isinstance(P, int):
        whattype = 'fixedp'
        xlab = N
    elif isinstance(P, list) and isinstance(N, int):
        whattype = 'fixedn'
        xlab = P
    else:
        raise ValueError("Either N or P should be a list and the other an integer")

    # Setup plot
    if ylab == 'Probability of rejection':
        # ax.set_ylim(0, 0.1)
        ax.axhline(y=alpha, color='black', linestyle='dotted', linewidth=1.25)
        
    # # Custom y-axis formatter for time plots
    # if ylab == 'Median log CPU time':
    #     ax.set_ylabel('Time (s)')  # Change the y-axis label to 'Time (s)'
    #     # Define a lambda function to format the y-axis ticks
    #     # It takes the tick value `y` and its position `pos`, and returns 2^y as a string
    #     formatter = FuncFormatter(lambda y, pos: f"{2**y:.2f}")
    #     ax.yaxis.set_major_formatter(formatter)

    # Process data
    plotmat = np.full((len(xlab), len(METHODS)), np.nan)
    for i, method in enumerate(METHODS):
        for x_idx, x in enumerate(xlab):
            n, p = (x, P) if whattype == 'fixedp' else (N, x)
            if ylab == 'Median log CPU time':
                filename = os.path.join(output_dir, f"{prefix}_n{n}p{p}_{method}_times.csv")
                try:
                    times = np.genfromtxt(
                        filename,
                        delimiter=",",
                        dtype=float,
                        filling_values=np.nan,  # replaces missing entries with NaN
                        skip_header=0,          # adjust if there's a header
                        invalid_raise=False     # prevents crash on bad rows
                        )
                    if len(times) < nsim:
                        times = np.concatenate([times, np.full(nsim - len(times), np.nan)])
                    # Filter out NaNs and check if there's any data left
                    valid_times = times[~np.isnan(times)]
                    if valid_times.size > 0:
                        log_times = np.log(valid_times+1)
                        plotmat[x_idx, i] = np.median(log_times)
                except (IOError, ValueError) as e:
                    print(f"Error doing file '{filename}': {e}")
                    pass
            else:
                filename = os.path.join(output_dir, f"{prefix}_n{n}p{p}_{method}_pvals.csv")
                try:
                    pvals = np.genfromtxt(
                        filename,
                        delimiter=",",
                        dtype=float,
                        filling_values=np.nan,  # replaces missing entries with NaN
                        skip_header=0,          # adjust if there's a header
                        invalid_raise=False     # prevents crash on bad rows
                        )
                    if pvals.ndim > 1:
                        pvals = pvals.flatten()

                    if len(pvals) < nsim:
                        pvals = np.concatenate([pvals, np.full(nsim - len(pvals), np.nan)])
                    # Check for empty pvals file before calculating
                    if pvals.size > 0 and not np.all(np.isnan(pvals)):
                        plotmat[x_idx, i] = (
                            np.mean(pvals < alpha) if ylab == 'Probability of rejection' else
                            ks_statistic(pvals) if ylab == 'KS statistic' else
                            compute_roc(oracle, pvals)[2]
                        )
                except (IOError, ValueError) as e:
                    print(f"Error doing file '{filename}': {e}")
                    pass

    # Check if plotmat is all nan
    if np.all(np.isnan(plotmat)):
        ax.axis('off')
        return
    
    # if ylab == 'AUROC':
    #     ax.set_ylim(min(0.75, plotmat.min()), 1)
    
    # Plot results
    markers = ['v', '^', '<', '>', 's', 'd', 'p']
    linestyles = ['--', ':', '-.', (0, (3, 1,1,1,1,1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (5, (10,3))]

    # Plot results for each method
    for i, method in enumerate(METHODS):
        # Get the display name for the method
        label_name = method_name_map.get(method, method)

        # Cycle through predefined styles for better discernibility
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        color = color_map.get(method, 'gray') # Use color_map, fallback to gray
        if 'medtree' in method:
            marker = 'o'
            linestyle = '-'

        # Use the x-axis data and the corresponding column from plotmat
        x_data = np.log(xlab) if ylab == 'Median log CPU time' else xlab
        y_data = plotmat[:, i]

        ax.plot(x_data, y_data,
                label=label_name,
                linestyle=linestyle,
                marker=marker,
                color=color,
                markersize=4,             # Consistent marker size
                # markerfacecolor='white',  # Hollow markers for clarity
                markeredgecolor=color,    # Edge color matches the line
                linewidth=1.25)            # Slightly thicker line

    if ylab == 'Median log CPU time':
        ax.plot(np.log(xlab), np.log(xlab) - min(np.log(xlab)), color='black', linestyle='dotted', linewidth=1.25)
        
    # Final plot formatting
    ax.set_ylabel(ylab)
    ax.set_xlabel("Sample size" if whattype == 'fixedp' else "Dimension of Z")
    ax.set_title(f"{'d = '+str(P) if whattype == 'fixedp' else 'n = '+str(N)}")
    if show_legend:
        # ax.legend(ncol=2, fontsize='small', loc='center', bbox_to_anchor=(0.675, 0.7), columnspacing=0.8)
        ax.legend(ncol=2, fontsize='small', columnspacing=0.8, frameon=False)
        # ax.legend(fontsize='small')
    
    if ylab == 'Median log CPU time':
        ax.set_xlabel("Sample size (log scale)" if whattype == 'fixedp' else "Dimension of Z (log scale)")
        ax.set_xticks(np.log(xlab))
        ax.set_xticklabels([str(x) for x in xlab])
    else:
        ax.set_xticks(xlab)
        tmp = [str(x) for x in xlab]
        tmp[1] = ''
        tmp[2] = ''
        ax.set_xticklabels(tmp)
    
    ax.tick_params(labelsize=9)
    # print(METHODS)
    # print(plotmat)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha='center')

    
def plot_time_lines(ax, output_dir, prefix, N, P, ylab, METHODS, nsim ,color_map, alpha=0.05, oracle=None, legend = True):
    """
    output_dir: Directory for output files
    prefix: 'null' or 'roc'
    N, p: 
    ylab: 'Median log CPU time'
    METHODS: List of method names
    """
    # Input validations
    if prefix not in {'null', 'roc'}:
        raise ValueError("prefix must be 'null' or 'roc'")
    if ylab not in {'Median log CPU time'}:
        raise ValueError("invalid ylab")

    # Determine fixed parameter type
    if isinstance(N, list) and isinstance(P, int):
        whattype = 'fixedp'
        xlab = N
    else:
        raise ValueError("N should be a list and p an integer")

    # Process data
    plotmat = np.full((len(xlab), len(METHODS)), np.nan)
    for i, method in enumerate(METHODS):
        for x_idx, x in enumerate(xlab):
            n, p = (x, P) if whattype == 'fixedp' else (N, x)
            if ylab == 'Median log CPU time':
                filename = os.path.join(output_dir, f"{prefix}_n{n}p{p}_{method}_times.csv")
                try:
                    times = np.genfromtxt(
                        filename,
                        delimiter=",",
                        dtype=float,
                        filling_values=np.nan,  # replaces missing entries with NaN
                        skip_header=0,          # adjust if there's a header
                        invalid_raise=False     # prevents crash on bad rows
                        )
                    if len(times) < nsim:
                        times = np.concatenate([times, np.full(nsim - len(times), np.nan)])
                    # Filter out NaNs and check if there's any data left
                    valid_times = times[~np.isnan(times)]
                    if valid_times.size > 0:
                        log_times = np.log(valid_times+1)
                        plotmat[x_idx, i] = np.median(log_times)
                except (IOError, ValueError) as e:
                    print(f"Error doing file '{filename}': {e}")
                    pass

    # Check if plotmat is all nan
    if np.all(np.isnan(plotmat)):
        ax.axis('off')
        return
    
    method_name_map = {
        'medtree_sttm_n15':'multiCMH',
        'medtree_sttm_n12':'multiCMH',
        'medtree_sttm_n10':'multiCMH',
        'medtree_sttm_n5':'multiCMH'
    }
    
    # Plot results
    markers = ['v', '<', 's', 'd']
    linestyles = ['--', '-.', (0, (3, 1, 1, 1)), (0, (1, 1))]
    for i, method in enumerate(METHODS):
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        if 'medtree' in method:
            marker = 'o'
            linestyle = '-'
        markerfacecolor = 'white'
        ax.plot(np.log(xlab), plotmat[:, i], label = method_name_map.get(method, method), linestyle=linestyle, marker=marker,
                linewidth = 1.25, markersize=4,
                color = color_map.get(method, 'gray'))

    ax.plot(np.log(xlab), np.log(xlab) - min(np.log(xlab)), color='gray', linestyle='dotted', linewidth=1.25)
    # Final plot formatting
    ax.set_ylabel(ylab)
    ax.set_xlabel("Sample size")
    ax.set_title(f"{'d = '+str(P)}")
    if legend:
        ax.legend(fontsize='small', frameon=False)
    
    if ylab == 'Median log CPU time':
        ax.set_xlabel("Sample size (thousands)")

    # ax.grid(True)
    ax.set_xticks(np.log(xlab))
    tmp = np.array([str(int(x/1000)) for x in xlab], dtype=object)
    tmp[1::2] = ''   # blanks every other label
    ax.set_xticklabels(tmp)
    # plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
    ax.set_ylim(-0.5, 9.5)