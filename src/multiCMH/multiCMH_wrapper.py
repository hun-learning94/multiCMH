# pip build -e .
from .cpp_modules import multiCMH_cpp_module

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr
import math
from typing import Literal

class multiCMH_output:
    """A class to hold the results of the multiCMH test."""
    def __init__(self, 
                 p_value: float, 
                 alpha: float, 
                 k1: int, 
                 k2: int,
                 x_dsc: bool,
                 y_dsc: bool,
                 sttm_n: int,
                 stt_method: Literal['ward', 'medtree'],
                 scr_all: int,
                 scr_mrg: int,
                 maxT: int,
                 x:np.ndarray, 
                 y:np.ndarray, 
                 Z:np.ndarray,
                 x_transformed: np.ndarray, 
                 y_transformed: np.ndarray, 
                 Z_transformed: np.ndarray, 
                 windows: np.ndarray, 
                 verbose: bool):
        self.alpha = alpha
        self.k1 = k1
        self.k2 = k2
        self.x_dsc = x_dsc
        self.y_dsc = y_dsc
        self.p_value = p_value
        self.sttm_n = sttm_n
        self.stt_method = stt_method
        self.scr_all = scr_all
        self.scr_mrg = scr_mrg
        self.maxT = maxT
        self.x = x
        self.y = y
        self.Z = Z
        self.x_transformed = x_transformed
        self.y_transformed = y_transformed
        self.Z_transformed = Z_transformed
        self.windows = windows
        self.verbose = verbose
    
    def __repr__(self):
        return f"multiCMH_output(p_value={self.p_value:.5f}, alpha={self.alpha:.1f}, k1={self.k1}, k2={self.k2})"
    
    def __str__(self):
        if self.p_value <= self.alpha:
            result = (
                f"The null is rejected with p-value={self.p_value:.3f} lower than alpha={self.alpha:.2f}.\n"
                f"Use .print_significant_windows() and .plot_significant_windows() for a detailed result.\n"
            )
        else:
            result = (
                f"The null is not rejected with p-value={self.p_value:.3f} higher than alpha={self.alpha:.2f}.\n"
            )
        params = (
            f"multiCMH parameters: "
            f"k1={self.k1}, k2={self.k2}, x_dsc={self.x_dsc}, y_dsc={self.y_dsc}, "
            f"stt_method={self.stt_method}, sttm_n={self.sttm_n}, maxT={self.maxT}, scr_all={self.scr_all}, scr_mrg={self.scr_mrg}, "
            f"verbose={self.verbose}"
        )
        return result + params
    
    def get_all_windows(self) -> pd.DataFrame:
        """
        Return full window-level results with confidence bounds. Requires verbose=True.
        """
        if not self.verbose:
            raise RuntimeError("To see details, please set verbose = true") 
        
        colnames = ['i', 'j', 'xl', 'xr', 'yl', 'yr', 'alp_sidak', 'pval_raw', 'cm_lodd', 'cm_lodd_sig', 'W_valid', 'Uw_valid', 'Lij_valid']
        df = pd.DataFrame(self.windows, columns = colnames)
        df['cm_lodd_lb95'] = df['cm_lodd'] - 1.96 * df['cm_lodd_sig']
        df['cm_lodd_ub95'] = df['cm_lodd'] + 1.96 * df['cm_lodd_sig']
        df = df[['i', 'j', 'xl', 'xr', 'yl', 'yr', 'alp_sidak', 'pval_raw', 'W_valid', 'Uw_valid', 'Lij_valid', 
                 'cm_lodd', 'cm_lodd_sig', 'cm_lodd_lb95', 'cm_lodd_ub95']]
        return df

    def get_significant_windows(self) -> pd.DataFrame:
        """
        Return subset of windows where alp_sidak > pval_raw. Optionally print the result. Requires verbose=True.
        """
        if not self.verbose:
            raise RuntimeError("To see details, please set verbose = true") 
        if not self.p_value < self.alpha:
            raise RuntimeError("There is no significant window: the null hypothesis is not rejected.") 
        
        df = self.get_all_windows()
        df = df[['i', 'j', 'xl', 'xr', 'yl', 'yr', 'alp_sidak', 'pval_raw', 'cm_lodd', 'cm_lodd_sig', 'cm_lodd_lb95', 'cm_lodd_ub95']]
        sigdf = df[df['alp_sidak'] > df['pval_raw']]
    
        return sigdf
    
    def print_significant_windows(self):
        """
        Prints a pandas DataFrame with aligned columns and formatted values.
        """
        if not self.verbose:
            raise RuntimeError("To see details, please set verbose = true") 
        if not self.p_value < self.alpha:
            raise RuntimeError("There is no significant window: the null hypothesis is not rejected.") 

        sigdf = self.get_significant_windows().copy()
        sigdf.insert(0, 'index', range(len(sigdf)))
        
        # sigdf.insert(sigdf.columns.get_loc('alp_sidak'), 
        #              'log10 alp', 
        #              np.log10(sigdf['alp_sidak']))
        # sigdf.drop(columns=['alp_sidak'], inplace=True)
        # sigdf.insert(sigdf.columns.get_loc('pval_raw'), 
        #              'log10 pval', 
        #              np.log10(sigdf['pval_raw']))
        # sigdf.drop(columns=['pval_raw'], inplace=True)
        
        xl_original, xr_original = np.quantile(self.x, sigdf['xl']), np.quantile(self.x, sigdf['xr'])
        yl_original, yr_original = np.quantile(self.y, sigdf['yl']), np.quantile(self.y, sigdf['yr'])
        xl_idx, xr_idx = sigdf.columns.get_loc('xl'), sigdf.columns.get_loc('xr')
        yl_idx, yr_idx = sigdf.columns.get_loc('yl'), sigdf.columns.get_loc('yr')
        sigdf.drop(columns=['xl', 'xr', 'yl', 'yr'], inplace=True)
        sigdf.insert(xl_idx, 'xl', xl_original)
        sigdf.insert(xr_idx, 'xr', xr_original)
        sigdf.insert(yl_idx, 'yl', yl_original)
        sigdf.insert(yr_idx, 'yr', yr_original)
        
        # Format header
        combined_headers = ['index', '(i,j)', '[xl, xr) x [yl, yr)']
        remaining_cols = [col for col in sigdf.columns if col not in ['index','i','j','xl','xr','yl','yr']]
        idx_col_width = 6
        ij_col_width = 7
        xy_col_width = 25
        regular_col_width = 12

        header_parts = [
            f"{combined_headers[0]:^{idx_col_width}}",
            f"{combined_headers[1]:^{ij_col_width}}",
            f"{combined_headers[2]:^{xy_col_width}}"
        ]
        header_parts += [f"{col:^{regular_col_width}}" for col in remaining_cols]
        header_string = f"| {' | '.join(header_parts)} |"
        print(header_string)
        print("-" * len(header_string))

        # Format each row
        def smart_format(val, tol=0.001):
            return f"{int(val)}" if abs(val - round(val)) < tol else f"{val:.2f}"
        
        for _, row in sigdf.iterrows():
            idx_val = f"{row['index']:.0f}"
            ij_combined = f"({row['i']:.0f},{row['j']:.0f})"
            xy_combined = (
                f"({smart_format(row['xl'])}, "
                f"{smart_format(row['xr'])}] x ("
                f"{smart_format(row['yl'])}, "
                f"{smart_format(row['yr'])}]"
            )
            formatted_row_parts = []
            for col in remaining_cols:
                item = row[col]
                if pd.isna(item):
                    formatted_row_parts.append(f"{'NaN':>{regular_col_width}}")
                elif item == float('inf'):
                    formatted_row_parts.append(f"{'inf':>{regular_col_width}}")
                elif 'alp' in col or 'pval' in col:
                    formatted_row_parts.append(f"{item:>{regular_col_width}.2e}")
                else:
                    formatted_row_parts.append(f"{item:>{regular_col_width}.2f}")
            row_string = f"| {idx_val:^{idx_col_width}} | {ij_combined:^{ij_col_width}} | {xy_combined:^{xy_col_width}} "
            row_string += " ".join([f"| {val}" for val in formatted_row_parts])
            row_string += " |"
            print(row_string)
    
    # helper function to extract stratum-specific information from a significant window
    def diagnose_significant_windows(self, idx: int, colnames=None, path=None, order=None):
        """
        Prints stratum-specific information for a chosen significant window,
        with floating-point values formatted to two decimal places.

        Args:
            idx (int): index into self.get_significant_windows().
            colnames (list[str] | None): optional names for Z columns.
            path (str | None): optional save path for the figure.
            order (list[str] | list[int] | None): optional user-specified order
                for rearranging the covariate plots.
                Examples:
                    order=['BMI', 'Age', 'Sex']
                    order=[2, 0, 1]
        """
        sigdf = self.get_significant_windows()
        if not 0 <= idx < sigdf.shape[0]:
            raise IndexError(f"Invalid index {idx}. Please choose an index between 0 and {sigdf.shape[0] - 1}.")

        # Get window information
        window_info = sigdf.iloc[idx]
        xl, xr = window_info['xl'], window_info['xr']
        yl, yr = window_info['yl'], window_info['yr']
        x_cut = (xl + xr) * 0.5
        y_cut = (yl + yr) * 0.5
        xl_original, xr_original = np.quantile(self.x, xl), np.quantile(self.x, xr)
        yl_original, yr_original = np.quantile(self.y, yl), np.quantile(self.y, yr)

        nobs_A = np.sum((self.x_transformed > xl) & (self.x_transformed <= xr) &
                        (self.y_transformed > yl) & (self.y_transformed <= yr))
        T_A = min(math.floor(nobs_A / self.sttm_n), self.maxT)
        output = multiCMH_cpp_module.singleCMH_detail(
            np.ascontiguousarray(self.x_transformed, dtype=np.float64),
            np.ascontiguousarray(self.y_transformed, dtype=np.float64),
            np.ascontiguousarray(self.Z, dtype=np.float64),
            x_cut, y_cut, int(T_A), int(self.scr_all), int(self.scr_mrg),
            {'ward': 1, 'medtree': 2}.get(self.stt_method)
        )

        Z_mean = output["Z_mean"]

        if colnames is not None:
            Z_cols = list(colnames)
        else:
            if isinstance(self.Z, pd.DataFrame):
                Z_cols = list(self.Z.columns)
            else:
                Z_cols = [f"Z{i+1}" for i in range(Z_mean.shape[1])]

        if len(Z_cols) != Z_mean.shape[1]:
            raise ValueError(
                f"Length of colnames ({len(Z_cols)}) must match number of Z columns ({Z_mean.shape[1]})."
            )

        # Reorder columns if requested
        if order is not None:
            if len(order) != len(Z_cols):
                raise ValueError(
                    f"Length of order ({len(order)}) must match number of Z columns ({len(Z_cols)})."
                )

            if all(isinstance(v, (int, np.integer)) for v in order):
                order_idx = list(order)
                if sorted(order_idx) != list(range(len(Z_cols))):
                    raise ValueError(
                        "When order is given as indices, it must be a permutation of "
                        f"0, 1, ..., {len(Z_cols)-1}."
                    )
            elif all(isinstance(v, str) for v in order):
                missing = [v for v in order if v not in Z_cols]
                if missing:
                    raise ValueError(f"Unknown column names in order: {missing}")
                order_idx = [Z_cols.index(v) for v in order]
            else:
                raise TypeError("order must be None, a list of column names, or a list of integer indices.")

            Z_mean = Z_mean[:, order_idx]
            Z_cols = [Z_cols[i] for i in order_idx]

        # Assemble the DataFrame
        sigdfidx = pd.concat([
            pd.Series(output["sttm_size"], name="stratum size"),
            pd.Series(output["slodds"], name="lodd"),
            pd.DataFrame(Z_mean, columns=Z_cols)
        ], axis=1)
        sigdfidx = sigdfidx.sort_values(by='lodd', ascending=True).reset_index(drop=True)

        fig, axes = plt.subplots(
            1, len(Z_cols),
            figsize=(2 * len(Z_cols) * 0.8, 2.75 * 0.8),
            sharey=True
        )
        if len(Z_cols) == 1:
            axes = [axes]

        for i, col in enumerate(Z_cols):
            x = sigdfidx[col].to_numpy().reshape(-1, 1)
            y = sigdfidx['lodd'].to_numpy()

            model = LinearRegression().fit(x, y)
            r, _ = pearsonr(
                (sigdfidx[col] - sigdfidx[col].mean()) / sigdfidx[col].std(),
                sigdfidx['lodd']
            )

            axes[i].axhline(y=0, color='black', linestyle='--', linewidth=1)
            axes[i].axhline(y=window_info['cm_lodd'], color='red', linestyle='-', linewidth=1)
            ymin, ymax = sigdfidx['lodd'].quantile(0.025), sigdfidx['lodd'].quantile(0.975)
            filtered = sigdfidx[(sigdfidx['lodd'] >= ymin) & (sigdfidx['lodd'] <= ymax)]

            sns.regplot(
                x=col, y='lodd', data=filtered, ax=axes[i],
                fit_reg=True, scatter=False
            )

            axes[i].set_xticks([])
            axes[i].set_xticklabels([])
            axes[i].set_xlabel(col, fontsize=10)

            if i == 0:
                axes[i].set_ylabel('Log Odds Ratio', fontsize=10)
            else:
                axes[i].set_ylabel('')

            axes[i].set_title(f'$\\rho$ = {r:.3f}', fontsize=10)
            axes[i].tick_params(axis='both', which='major', labelsize=10)

        def smart_format(val, tol=0.001):
            return f"{int(val)}" if abs(val - round(val)) < tol else f"{val:.2f}"

        fig.suptitle(
            f"Partition ({window_info['i']:.0f},{window_info['j']:.0f}), "
            f"Window ({smart_format(xl_original)}, {smart_format(xr_original)}] x "
            f"[{smart_format(yl_original)}, {smart_format(yr_original)}]",
            fontsize=11,
            y=0.99
        )
        plt.tight_layout(pad=0.0, w_pad=0.1, h_pad=0.5, rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(wspace=0.05, right=0.98)

        if path is not None:
            plt.savefig(path, dpi=300)
        plt.show()
        

    # helper functions for plotting
    def _plot_window(self, ax, i, j, norm, cmap, color = 'p-value'):
        df = self.get_all_windows()
        df_ij = df[(df['i'] == i) & (df['j'] == j)].dropna(subset=['pval_raw'])
        for _, row in df_ij.iterrows():
            edgecolor = 'white'
            linewidth = 1
            # if row['pval_raw'] < row['alp_sidak']:
            #     edgecolor = 'red'
            #     linewidth = 2
            if color == 'p-value':
                facecolor=cmap(norm(row['pval_raw']))
            elif color == 'cm_lodd':
                facecolor=cmap(norm(row['cm_lodd']))
            if row['pval_raw'] < row['alp_sidak']:
                rect = patches.Rectangle(
                    (row['xl'], row['yl']),
                    row['xr'] - row['xl'],
                    row['yr'] - row['yl'],
                    linewidth=linewidth,
                    edgecolor=edgecolor,
                    facecolor=facecolor
                )
                ax.add_patch(rect)
        ax.set_xlim((-0.05, 1.05))
        ax.set_ylim((-0.05, 1.05))
        ax.set_title(f'({int(i)},{int(j)})', fontsize=10)
        ax.set_aspect('equal')
        xticks = yticks = [0, 0.25, 0.5, 0.75, 1]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([f"{q:.1f}" for q in np.quantile(self.x, xticks)], fontsize=9)
        ax.set_yticklabels([f"{q:.1f}" for q in np.quantile(self.y, yticks)], fontsize=9)
        
    def _scatter_transformed(self, ax):
        ax.scatter(self.x_transformed, self.y_transformed, s=1, c='black', alpha=0.5)
        ax.set_aspect('equal')
        ax.set_xlim((-0.05, 1.05))
        ax.set_ylim((-0.05, 1.05))
        xticks = yticks = [0, 0.25, 0.5, 0.75, 1]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([f"{q:.1f}" for q in np.quantile(self.x, xticks)], fontsize=9)
        ax.set_yticklabels([f"{q:.1f}" for q in np.quantile(self.y, yticks)], fontsize=9)
        
    def _scatter_original(self, ax):
        x_min, x_max = np.nanmin(self.x), np.nanmax(self.x)
        y_min, y_max = np.nanmin(self.y), np.nanmax(self.y)
        x_standardized = (self.x - x_min) / (x_max - x_min)
        y_standardized = (self.y - y_min) / (y_max - y_min)
        ax.scatter(x_standardized, y_standardized, s=1, c="black", alpha=0.5)
        x_tick_locs = ax.get_xticks()
        y_tick_locs = ax.get_yticks()
        x_original_labels = x_tick_locs * (x_max - x_min) + x_min
        y_original_labels = y_tick_locs * (y_max - y_min) + y_min
        x_labels_formatted = [f'{val:.1f}' for val in x_original_labels]
        y_labels_formatted = [f'{val:.1f}' for val in y_original_labels]
        ax.set_xticks(x_tick_locs)
        ax.set_yticks(y_tick_locs)
        ax.set_xticklabels(x_labels_formatted, fontsize=9)
        ax.set_yticklabels(y_labels_formatted, fontsize=9)
        ax.set_xlim((-0.05, 1.05))
        ax.set_ylim((-0.05, 1.05))
        
    def plot_significant_windows(self, color = 'p-value', path = None):
        if not self.p_value < self.alpha:
            raise RuntimeError("There is no significant window: the null hypothesis is not rejected.") 
        sigdf = self.get_significant_windows()
        siglvs = sigdf[['i','j']].drop_duplicates()
        num_siglvs = siglvs.shape[0]
        
        ntotal = 2 + num_siglvs
        ncol_real = 5
        ncol = ncol_real + 1
        nrow = ((ntotal - 1) // ncol_real) + 1
        
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*2, nrow*2), gridspec_kw={'width_ratios': [1, 1,1,1,1, 0.1]})
        axes_flat=axes.flatten()
        if color == 'p-value':
            # norm = colors.LogNorm(vmin=min(1e-10, 
            #                                sigdf['pval_raw'][np.isfinite(sigdf['pval_raw'])].min()), 
            #                       vmax=1) 
            norm = colors.SymLogNorm(linthresh=1e-10, linscale=1.0, vmin=1e-10, 
                                     vmax=sigdf['alp_sidak'][np.isfinite(sigdf['alp_sidak'])].max())
            cmap = plt.get_cmap('Greens_r')
            label = 'Uncorrected p-value'
        elif color == 'cm_lodd':
            finite_vals = sigdf['cm_lodd'][np.isfinite(sigdf['cm_lodd'])]
            vmin = finite_vals.min()
            vmax = finite_vals.max()
            vbound = max(abs(vmin), abs(vmax))
            norm = colors.TwoSlopeNorm(vmin=-vbound, vcenter=0, vmax=vbound)
            cmap = plt.get_cmap('bwr_r')  # blue → white → red, or 'seismic', 'coolwarm', 'RdBu'
            label = 'Common log odds ratio'
        else:
            raise ValueError("color must be p-value or cm_lodd")
        

        plots_drawn = 0
        for t in range(ncol*nrow):
            if t == 0:
                self._scatter_original(axes_flat[t])
                axes_flat[t].set_title("Original", fontsize=10)
                axes_flat[t].set_aspect('equal')
            if t == 1:
                self._scatter_transformed(axes_flat[t])
                axes_flat[t].set_title("Transformed", fontsize=10)
                axes_flat[t].set_aspect('equal')
            if t > 1:
                if (t+1) % ncol == 0:
                    axes_flat[t].axis('off')
                else:
                    try:
                        i = siglvs['i'].iloc[plots_drawn]
                        j = siglvs['j'].iloc[plots_drawn]
                        self._plot_window(axes_flat[t], i, j, norm, cmap, color)
                        plots_drawn += 1
                    except IndexError:
                        axes_flat[t].axis('off')

        axes_flat[t].axis('on')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=axes_flat[-1])
        cbar.set_label(label)

        plt.tight_layout(pad=0., w_pad=0, h_pad=0.5)
        if path is not None:
            plt.savefig(path, dpi=300)
        plt.show()
        
    
    

def multiCMH(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    x_dsc: bool = False,
    y_dsc: bool = False,
    k1: int = None,
    k2: int = None,
    alp: float = 0.05,
    sttm_n: int = 12,
    stt_method : str = 'medtree',
    scr_all: int = 20,
    scr_mrg: int = 10,
    maxT: int = 200,
    verbose: bool = False
) -> multiCMH_output:
    """
    Python wrapper for the multi-scale CMH test.

    Performs input validation on NumPy arrays and calls the C++ function.

    Args:
        x (np.ndarray): 1D array of x values.
        y (np.ndarray): 1D array of y values.
        Z (np.ndarray): 2D array of Z values (n_obs x n_dims).
        x_dsc (bool): Is x a discrete variable?
        y_dsc (bool): Is y a discrete variable?
        k1 (int): Maximum partition level for x.
        k2 (int): Maximum partition level for y.
        alp (float): Significance level for Sidak correction.
        sttm_n (int): Desired number of samples per stratum.
        stt_method (str): ward or medtree. default medtree
        scr_all (int): Minimum total count for a window.
        scr_mrg (int): Minimum marginal count for a window.
        maxT (int): maximum number of stratifications per window. default 200
        verbose (bool): whether or not to return all windows p-values, alphas and more

    Returns:
        multiCMH_output: An object containing the computed p-value.
    
    Raises:
        TypeError: If inputs are not of the expected type.
        ValueError: If input arrays have incompatible shapes or values.
        RuntimeError: If the C++ function encounters an error.
    """
    # input validation
    if not all(isinstance(arr, np.ndarray) for arr in [x, y, Z]):
        raise TypeError("Input data x, y, and Z must be numpy.ndarrays.")
    if x.ndim != 1 or y.ndim != 1 or Z.ndim != 2:
        raise ValueError("x and y must be (n, ), Z must be 2D (n, p)")
    if not (x.shape[0] == y.shape[0] == Z.shape[0]):
        raise ValueError("x, y, and Z must have the same n")
    if stt_method not in ('ward', 'medtree'):
        raise ValueError("stt_method must be 'ward' or 'medtree'")

    # double check x, y, Z are readable by pybind11
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    
    # determine the depths k1, k2. If x and/or y is discrete, k1 and k2 is computed inside the cpp function
    # such that the resulting parition contains only one unique value in each partition
    if k1 is None:
        k1 = min(math.floor(math.log2(x.shape[0] / scr_mrg)), 7)
    if k2 is None:
        k2 = min(math.floor(math.log2(y.shape[0] / scr_mrg)), 7)
    max_k = 7 # hard-coded maximum depth. 7 or 8 is suggested
    k1 = min(k1, max_k)
    k2 = min(k2, max_k)
    
    # adjust scr_mrg, scr_all if x and/or y is discrete
    if x_dsc == True:
        _, cnts = np.unique(x, return_counts=True)
        min_cnt = int(np.min(cnts))
        if scr_mrg > min_cnt:
            scr_mrg = min_cnt
            scr_all = min(2 * scr_mrg, scr_all)
    if y_dsc == True:
        _, cnts = np.unique(y, return_counts=True)
        min_cnt = int(np.min(cnts))
        if scr_mrg > min_cnt:
            scr_mrg = min_cnt
            scr_all = min(2 * scr_mrg, scr_all)
            
    stt_method_int = {'ward': 1, 'medtree': 2}.get(stt_method)
        
    # 2. Call the C++ Function
    try:
        output = multiCMH_cpp_module.multiCMH_cpp(
            x, y, Z, x_dsc, y_dsc, k1, k2, alp, sttm_n, stt_method_int, scr_all, scr_mrg, maxT, verbose
        )
    except Exception as e:
        # Re-raise C++ exceptions for better debugging
        raise RuntimeError(f"Error from C++ function multiCMH_cpp: {e}") from e

    # 3. Return a user-friendly object
    return multiCMH_output(
        p_value = output["p_value"],
        alpha = alp,
        k1 = output["k1"],
        k2 = output["k2"],
        x_dsc = x_dsc,
        y_dsc = y_dsc,
        sttm_n = sttm_n,
        stt_method = stt_method,
        scr_all = scr_all,
        scr_mrg = scr_mrg,
        maxT = maxT,
        x = x,
        y = y,
        Z = Z,
        x_transformed = output["x_transformed"],
        y_transformed = output["y_transformed"],
        Z_transformed = output["Z_transformed"],
        windows = output["windows"],
        verbose = verbose)


def ward_fastcluster(X: np.ndarray, K: int) -> np.ndarray:
    """
    Python wrapper for the Ward linkage agglomerative clustering function.

    Args:
        X (np.ndarray): A 2D array of data points (n_obs, n_dims).
        K (int): The number of clusters to form.

    Returns:
        np.ndarray: A 1D array of integers denoting cluster assignment (n_obs, ).

    Raises:
        TypeError: If X is not a NumPy array.
        ValueError: If X is not a 2D array or K is invalid.
        RuntimeError: If the C++ function encounters an error.
    """
    # 1. Input Validation for ward_fastcluster
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a NumPy array.")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array.")
    if not isinstance(K, int) or K <= 0:
        raise ValueError("K must be a positive integer.")

    # 2. Call the C++ Function
    try:
        labels = multiCMH_cpp_module.Ward_fastcluster(X, K)
    except Exception as e:
        raise RuntimeError(f"Error from C++ function Ward_fastcluster: {e}") from e

    # 3. Return the result directly
    return labels


def medtree_cluster(X: np.ndarray, K: int) -> np.ndarray:
    """
    Python wrapper for the median tree (KD tree) clustering function.

    Args:
        X (np.ndarray): A 2D array of data points (n_obs, n_dims).
        K (int): The number of clusters to form.

    Returns:
        np.ndarray: A 1D array of integers denoting cluster assignment (n_obs, ).

    Raises:
        TypeError: If X is not a NumPy array.
        ValueError: If X is not a 2D array or K is invalid.
        RuntimeError: If the C++ function encounters an error.
    """
    # 1. Input Validation for MedTree_cluster
    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a NumPy array.")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array.")
    if not isinstance(K, int) or K <= 0:
        raise ValueError("K must be a positive integer.")

    # 2. Call the C++ Function
    try:
        labels = multiCMH_cpp_module.MedTree_cluster(X, K)
    except Exception as e:
        raise RuntimeError(f"Error from C++ function MedTree_cluster: {e}") from e

    # 3. Return the result directly
    return labels

# Example Usage
# to test this, at the root of the project (where setup.py is at), run
# python -m src.multiCMH.multiCMH_wrapper
if __name__ == '__main__':
    # --- Example for multiCMH ---
    print("--- Testing multiCMH wrapper ---")
    n_samples = 200
    n_dims = 3
    x = np.random.rand(n_samples)
    y = np.random.rand(n_samples)
    Z = np.random.rand(n_samples, n_dims)

    try:
        print("\nRunning a valid multiCMH test...")
        result = multiCMH(x=x,y=y,Z=Z)
        print(f"Test passed. Result: {result}")
    except Exception as e:
        print(f"An unexpected error occurred during multiCMH test: {e}")

    # --- Example for Ward_fastcluster ---
    print("\n--- Testing ward_fastcluster wrapper ---")
    n_points = 50
    n_features = 4
    X = np.random.rand(n_points, n_features)
    n_clusters = 3

    try:
        print(f"\nRunning a valid Ward clustering test with K={n_clusters}...")
        labels = ward_fastcluster(X, n_clusters)
        print("Labels generated successfully.")
    except Exception as e:
        print(f"An unexpected error occurred during Ward clustering test: {e}")
        
    # --- Example for medtree_cluster ---
    print("\n--- Testing medtree_cluster wrapper ---")
    n_points = 50
    n_features = 4
    X = np.random.rand(n_points, n_features)
    n_clusters = 3

    try:
        print(f"\nRunning a valid medtree clustering test with K={n_clusters}...")
        labels = medtree_cluster(X, n_clusters)
        print("Labels generated successfully.")
    except Exception as e:
        print(f"An unexpected error occurred during medtree clustering test: {e}")
