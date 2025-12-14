from __future__ import print_function, division

import os
import numpy as np
from numpy.random import normal
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.stats import pearsonr, spearmanr, ttest_ind
from scipy.linalg import inv, solve, cholesky
from numba import jit
from concurrent.futures import ProcessPoolExecutor
import warnings

from calculate_rate import transduction_function
from helper_functions import calculate_periodogram, matlab_smooth, plt_filled_std, get_network_configuration

# Set random seed for reproducibility
np.random.seed(42)


# ============================================================================
# CONNECTIVITY DATA
# ============================================================================

def get_macaque_connectivity(path="data/macaque_connectivity.npz"):
    """
    Get the macaque connectivity data.
    """
    data = np.load(path, allow_pickle=True)
    area_names = data["area_names"]
    fln = data["fln"]
    sln = data["sln"]
    wiring = data["wiring_dist"]

    return area_names, fln, sln, wiring


def get_roi_subset(areas):
    """
    Returns the ROI subset used in Bastos et al. (2015) / Kennedy-Fries analysis.
    These 8 areas: V1, V2, V4, DP, 8m, 8l, TEO, 7A
    """
    # Indices in the 30-area list
    roi_names = ['V1', 'V2', 'V4', 'DP', 'MT', '8m', '8l', 'TEO', '7A', 'TEpd', '46d']
    roi_indices = [list(areas).index(str(name)) for name in roi_names]
    return roi_names, roi_indices


# ============================================================================
# NETWORK PARAMETERS
# ============================================================================

def get_largescale_parameters(Nareas=30, dt=2e-4):
    """
    Get parameters for the large-scale network simulation.
    
    Parameters
    ----------
    Nareas : int
        Number of cortical areas
    dt : float
        Time step in seconds
        
    Returns
    -------
    par : dict
        Dictionary containing all simulation parameters
    """
    tau, sig, J, _, _ = get_network_configuration('largescale', noconns=False)
    
    # Compute time step factors
    tstep = (dt / tau)
    tstep2 = np.sqrt(dt * sig * sig / tau)
    
    # Binning for analysis
    binx = 20
    eta = 0.2  # Recording depth (0=L5/6 only, 1=L2/3 only)
    
    par = {
        'dt': dt,
        'tau': tau,
        'sig': sig,
        'J': J,
        'tstep': tstep,
        'tstep2': tstep2,
        'binx': binx,
        'eta': eta,
        'Nareas': Nareas
    }
    
    return par


def build_largescale_connectivity(fln_mat, sln_mat, wiring, par, G=1.1):
    """
    Build the large-scale connectivity matrices.
    
    Parameters
    ----------
    fln_mat : ndarray
        FLN connectivity matrix
    sln_mat : ndarray 
        SLN laminar specificity matrix
    wiring : ndarray
        Wiring distances in mm
    par : dict
        Parameter dictionary
    G : float
        Global coupling gain
        
    Returns
    -------
    Wff : ndarray
        Feedforward connectivity weights
    Wfb : ndarray
        Feedback connectivity weights  
    delays : ndarray
        Axonal propagation delays in dt units
    s : ndarray
        Selectivity matrix for FB projections
    """
    Nareas = par['Nareas']
    dt = par['dt']
    
    # Range compression for connection weights (as in Matlab code)
    fln_compressed = 1.2 * np.power(fln_mat, 0.30)
    
    # Feedforward and feedback matrices
    Wff = fln_compressed * sln_mat
    Wfb = fln_compressed * (1 - sln_mat)
    Wff = np.ascontiguousarray(Wff)
    Wfb = np.ascontiguousarray(Wfb)
    
    # Normalize by total input to each area
    for i in range(Nareas):
        normff = np.sum(Wff[i, :])
        normfb = np.sum(Wfb[i, :])
        if normff > 0:
            Wff[i, :] = G * Wff[i, :] / normff
        if normfb > 0:
            Wfb[i, :] = G * Wfb[i, :] / normfb
    
    # Compute delays from wiring distances
    # v = 1.5 m/s = 1500 mm/s
    velocity = 1500  # mm/s
    delays = np.round(1 + wiring / (velocity * dt)).astype(int)
    delays = np.ascontiguousarray(delays)
    
    # Selectivity matrix for FB projections (proportion targeting different layers)
    s = np.ascontiguousarray(0.1 * np.ones((Nareas, Nareas)))
    
    return Wff, Wfb, delays, s


# ============================================================================
# SIMULATION
# ============================================================================

def largescale_simulation(par, Wff, Wfb, delays, s, Iext, tstop=200, transient=20, clamp_mask=None, seed=None):
    """
    Run a single trial of the large-scale network simulation with delays.
    
    Parameters
    ----------
    par : dict
        Parameter dictionary
    Wff : ndarray
        Feedforward connectivity
    Wfb : ndarray
        Feedback connectivity
    delays : ndarray
        Axonal delays in dt units
    s : ndarray
        Selectivity matrix
    Iext : ndarray (4, Nareas)
        External input to each population in each area
    tstop : float
        Simulation length in seconds
    transient : float
        Transient period to discard in seconds
    clamp_mask : ndarray (Nareas,) or None
        Boolean mask indicating which areas to clamp to zero
    seed : int or None
        RNG seed for reproducible noise
        
    Returns
    -------
    rate : ndarray (4, Nt, Nareas)
        Firing rates for all populations and areas
    """
    dt = par['dt']
    J = par['J']
    tau = par['tau']
    tstep = par['tstep']
    tstep2 = par['tstep2']
    Nareas = par['Nareas']
    
    # Number of time points
    Nt = int(round(tstop / dt))
    
    # Initialize arrays
    rate = np.zeros((4, Nt, Nareas))
    if seed is None:
        xi = np.random.randn(4, Nt, Nareas)  # Noise
    else:
        rng = np.random.default_rng(seed)
        xi = rng.standard_normal((4, Nt, Nareas))
    
    # Build selectivity matrices for FB projections
    compls = 1 - s
    Wfbe1 = s * Wfb  # To L2/3E
    Wfbe2 = compls * Wfb  # To L5/6E
    
    # Initial conditions (random between 0 and 10 Hz)
    rate[:, 0, :] = 5 * (1 + np.tanh(2 * xi[:, 0, :]))
    
    # Background input
    inputbg = np.zeros((4, Nareas))
    
    # Minimum time before we can use delays (need > max_delay time steps)
    # If time > 0.2 (little transient >59ms)
    delay_transient_steps = int(0.2 / dt)  # ~1000 steps at dt=0.2ms

    # Flattened delay structures to avoid Python nested loops
    delay_flat = delays.ravel()
    pre_idx_flat = np.tile(np.arange(Nareas), Nareas)
    dr_flat1 = np.zeros_like(delay_flat, dtype=np.float64)
    dr_flat3 = np.zeros_like(delay_flat, dtype=np.float64)
    drate1 = np.zeros((Nareas, Nareas))
    drate3 = np.zeros((Nareas, Nareas))

    # Precompute broadcastable time-step vectors
    tstep_vec = tstep.reshape(4, 1)
    tstep2_vec = tstep2.reshape(4, 1)
    
    # Main simulation loop
    for t in range(1, Nt):
        # Current rates
        irate = rate[:, t-1, :]
        
        # Get delayed rates for interareal connections
        # Only compute delayed inputs after sufficient time has passed
        if t > delay_transient_steps:
            valid_steps = t - delay_flat
            valid_mask = valid_steps >= 0

            # Fetch delayed rates in one vectorized pass
            dr_flat1.fill(0.0)
            dr_flat3.fill(0.0)
            if np.any(valid_mask):
                idx = valid_steps[valid_mask].astype(int)
                pre_idx = pre_idx_flat[valid_mask]
                dr_flat1[valid_mask] = rate[0, idx, pre_idx]  # L2/3E
                dr_flat3[valid_mask] = rate[2, idx, pre_idx]  # L5/6E

            drate1[:] = dr_flat1.reshape(Nareas, Nareas)
            drate3[:] = dr_flat3.reshape(Nareas, Nareas)
        else:
            drate1.fill(0.0)
            drate3.fill(0.0)
        
        # Compute total input
        total_input = inputbg + Iext + J @ irate
        
        # Interareal FF projections (L2/3E -> L2/3E)
        total_input[0, :] += np.einsum('ij,ij->i', drate1, Wff)
        
        # Interareal FB projections (L5/6E -> multiple targets)
        total_input[0, :] += np.einsum('ij,ij->i', drate3, Wfbe1)     # -> L2/3E
        total_input[1, :] += 0.5 * np.einsum('ij,ij->i', drate3, Wfb) # -> L2/3I
        total_input[2, :] += np.einsum('ij,ij->i', drate3, Wfbe2)     # -> L5/6E
        total_input[3, :] += 0.5 * np.einsum('ij,ij->i', drate3, Wfb) # -> L5/6I
        
        # Transfer function
        transfer = transduction_function(total_input)
        
        # Update rates
        rate[:, t, :] = irate + tstep_vec * (-irate + transfer) + tstep2_vec * xi[:, t, :]

        # Apply lesion clamping if specified
        if clamp_mask is not None:
            rate[:, t, clamp_mask] = 0.0
    
    return rate


def run_largescale_trials(par, Wff, Wfb, delays, s, Iext, tstop=200, transient=20, ntrials=30, clamp_mask=None, base_seed=42):
    """
    Run multiple trials of the large-scale simulation.
    
    Parameters
    ----------
    par : dict
        Parameter dictionary
    Wff, Wfb, delays, s : arrays
        Connectivity matrices
    Iext : ndarray
        External input
    tstop : float
        Trial length in seconds
    transient : float
        Transient to discard
    ntrials : int
        Number of trials
    clamp_mask : ndarray or None
        Areas to clamp (None = baseline, array = lesioned)
    base_seed : int or None
        Base seed for reproducible per-trial RNG (seed+trial_idx)
        
    Returns
    -------
    X : ndarray (Nareas, Nt_binned, ntrials)
        Combined L2/3+L5/6 activity for Granger analysis
    X2 : ndarray (Nareas, Nt_binned, ntrials)
        L2/3E activity
    X5 : ndarray (Nareas, Nt_binned, ntrials)
        L5/6E activity
    """
    dt = par['dt']
    binx = par['binx']
    eta = par['eta']
    Nareas = par['Nareas']
    
    # Calculate binned time points
    nobs_full = int(round((tstop - transient) / dt))
    nobs_binned = nobs_full // binx
    t0_idx = int(round(transient / dt))
    
    # Initialize arrays
    X = np.zeros((Nareas, nobs_binned, ntrials))
    X2 = np.zeros((Nareas, nobs_binned, ntrials))
    X5 = np.zeros((Nareas, nobs_binned, ntrials))
    
    for trial in range(ntrials):
        print(f'    Running trial {trial+1} of {ntrials}...')

        seed = None if base_seed is None else base_seed + trial
        rate_local = largescale_simulation(par, Wff, Wfb, delays, s, Iext, tstop, transient, clamp_mask, seed=seed)

        X_loc = np.zeros((Nareas, nobs_binned))
        X2_loc = np.zeros((Nareas, nobs_binned))
        X5_loc = np.zeros((Nareas, nobs_binned))

        for area in range(Nareas):
            r2 = rate_local[0, t0_idx::binx, area][:nobs_binned]
            r5 = rate_local[2, t0_idx::binx, area][:nobs_binned]
            X2_loc[area, :] = r2
            X5_loc[area, :] = r5
            X_loc[area, :] = eta * r2 + (1 - eta) * r5

        X[:, :, trial] = X_loc
        X2[:, :, trial] = X2_loc
        X5[:, :, trial] = X5_loc    
    
    return X, X2, X5


# ==============================================================================
# LESIONS AND FIRE RATE ANALYSIS
# ==============================================================================

def apply_lesion(Wff, Wfb, delays, s, Iext, lesion_areas, lesion_type='complete'):
    """
    Apply lesion to the network by modifying connectivity and/or activity.
    
    Parameters
    ----------
    Wff : ndarray (Nareas, Nareas)
        Feedforward connectivity matrix
    Wfb : ndarray (Nareas, Nareas)
        Feedback connectivity matrix
    delays : ndarray (Nareas, Nareas)
        Delay matrix in time steps
    s : ndarray (Nareas, Nareas)
        Short-range connectivity matrix
    Iext : ndarray (4, Nareas)
        External input currents
    lesion_areas : list of int
        Indices of areas to lesion
    lesion_type : str
        Type of lesion:
        - 'complete': Remove all activity and connectivity
        - 'activity_only': Clamp activity to 0, keep connections
        - 'output_loss': Remove outgoing connections only
        - 'input_loss': Remove incoming connections only
    
    Returns
    -------
    Wff_lesion, Wfb_lesion, delays_lesion, s_lesion, Iext_lesion, clamp_mask
        Modified connectivity matrices and a mask indicating which areas to clamp
    """
    # Make copies to avoid modifying originals
    Wff_lesion = Wff.copy()
    Wfb_lesion = Wfb.copy()
    delays_lesion = delays.copy()
    s_lesion = s.copy()
    Iext_lesion = Iext.copy()
    
    Nareas = Wff.shape[0]
    clamp_mask = np.zeros(Nareas, dtype=bool)
    
    for area_idx in lesion_areas:
        if lesion_type == 'complete':
            # Remove all activity
            clamp_mask[area_idx] = True
            Iext_lesion[:, area_idx] = 0
            
            # Remove all outgoing connections
            Wff_lesion[area_idx, :] = 0
            Wfb_lesion[area_idx, :] = 0
            s_lesion[area_idx, :] = 0
            
            # Remove all incoming connections
            Wff_lesion[:, area_idx] = 0
            Wfb_lesion[:, area_idx] = 0
            s_lesion[:, area_idx] = 0
            
        elif lesion_type == 'activity_only':
            # Only clamp activity, leave connections intact
            clamp_mask[area_idx] = True
            Iext_lesion[:, area_idx] = 0

        elif lesion_type == 'input_loss':
            # Remove incoming connections only (deafferentation)
            Wff_lesion[:, area_idx] = 0
            Wfb_lesion[:, area_idx] = 0
            s_lesion[:, area_idx] = 0
            
        elif lesion_type == 'output_loss':
            # Remove outgoing connections only (white matter lesion)
            Wff_lesion[area_idx, :] = 0
            Wfb_lesion[area_idx, :] = 0
            s_lesion[area_idx, :] = 0
    
    return Wff_lesion, Wfb_lesion, delays_lesion, s_lesion, Iext_lesion, clamp_mask


def lesion_rate_analysis(X_baseline, X_lesion, area_names, lesion_areas):
    """
    Analyze changes in mean firing rates after lesion.
    
    Parameters
    ----------
    X_baseline : ndarray (Nareas, Nt, ntrials)
        Baseline activity
    X_lesion : ndarray (Nareas, Nt, ntrials)
        Lesioned activity
    area_names : list of str
        Area names
    lesion_areas : list of int
        Indices of lesioned areas
    
    Returns
    -------
    dict with keys:
        - mean_baseline : ndarray (Nareas,)
        - mean_lesion : ndarray (Nareas,)
        - percent_change : ndarray (Nareas,)
        - pvalues : ndarray (Nareas,)
        - significant : ndarray (Nareas,) bool
    """
    Nareas = X_baseline.shape[0]
    
    # Compute mean rates across time and trials
    mean_baseline = np.mean(X_baseline, axis=(1, 2))
    mean_lesion = np.mean(X_lesion, axis=(1, 2))
    
    # Percent change
    percent_change = 100 * (mean_lesion - mean_baseline) / (mean_baseline + 1e-10)
    
    # Statistical test (t-test across trials)
    pvalues = np.zeros(Nareas)
    for i in range(Nareas):
        baseline_trials = np.mean(X_baseline[i, :, :], axis=0)
        lesion_trials = np.mean(X_lesion[i, :, :], axis=0)
        _, pvalues[i] = ttest_ind(baseline_trials, lesion_trials)
    
    significant = pvalues < 0.05
    
    return {
        'mean_baseline': mean_baseline,
        'mean_lesion': mean_lesion,
        'percent_change': percent_change,
        'pvalues': pvalues,
        'significant': significant
    }


# ============================================================================
# POWER ANALYSIS
# ============================================================================

def largescale_power_analysis(X2, X5, par, transient=0):
    """
    Compute alpha and gamma power for all areas.
    
    Parameters
    ----------
    X2 : ndarray (Nareas, Nt, ntrials)
        L2/3E activity
    X5 : ndarray (Nareas, Nt, ntrials)
        L5/6E activity
    par : dict
        Parameters
    transient : float
        Transient to skip (in seconds)
        
    Returns
    -------
    results : dict
        Dictionary with gamma_power, alpha_power
        gamma_power : ndarray (Nareas, ntrials)
            Gamma power per area
        alpha_power : ndarray (Nareas, ntrials)
            Alpha power per area
    """

    Nareas, Nt, ntrials = X2.shape
    dt = par['dt'] * par['binx']  # Binned dt
    fs = 1 / dt
    
    gamma_power = np.zeros((Nareas, ntrials))
    alpha_power = np.zeros((Nareas, ntrials))
    
    for trial in range(ntrials):
        for area in range(Nareas):
            # L2/3 for gamma
            pxx2, fxx2 = calculate_periodogram(X2[area, :, trial], 0, dt)
            gamma_idx = np.where((fxx2 >= 30) & (fxx2 <= 70))[0]
            if len(gamma_idx) > 0:
                gamma_power[area, trial] = np.max(pxx2[gamma_idx])
            
            # L5/6 for alpha
            pxx5, fxx5 = calculate_periodogram(X5[area, :, trial], 0, dt)
            alpha_idx = np.where((fxx5 >= 4) & (fxx5 <= 18))[0]
            if len(alpha_idx) > 0:
                alpha_power[area, trial] = np.max(pxx5[alpha_idx])
    
    return {
        'gamma_power': gamma_power,
        'alpha_power': alpha_power
    }


def lesion_power_analysis(X2_baseline, X5_baseline, X2_lesion, X5_lesion, par):
    """
    Analyze changes in oscillatory power after lesion.
    
    Parameters
    ----------
    X2_baseline, X5_baseline : ndarray (Nareas, Nt_binned, ntrials)
        Baseline L2/3 and L5/6 activity (binned)
    X2_lesion, X5_lesion : ndarray (Nareas, Nt_binned, ntrials)
        Lesioned L2/3 and L5/6 activity (binned)
    par : dict
        Network parameters
    
    Returns
    -------
    dict with power analysis results
    """
    dt = par['dt']
    binx = par['binx']
    Nareas = par['Nareas']
    ntrials = X2_baseline.shape[2]
    
    # Sampling frequency for binned data
    fs = 1.0 / (dt * binx)
    
    # Initialize arrays
    gamma_power_base = np.zeros((Nareas, ntrials))
    alpha_power_base = np.zeros((Nareas, ntrials))
    gamma_power_les = np.zeros((Nareas, ntrials))
    alpha_power_les = np.zeros((Nareas, ntrials))
    
    print('    Computing baseline power spectra...')
    for trial in range(ntrials):
        for area in range(Nareas):
            # Baseline - L2/3 gamma
            fxx2, pxx2 = signal.periodogram(X2_baseline[area, :, trial], fs=fs)
            gamma_idx = np.where((fxx2 >= 30) & (fxx2 <= 70))[0]
            gamma_power_base[area, trial] = np.max(pxx2[gamma_idx]) if len(gamma_idx) > 0 else 0
            
            # Baseline - L5/6 alpha
            fxx5, pxx5 = signal.periodogram(X5_baseline[area, :, trial], fs=fs)
            alpha_idx = np.where((fxx5 >= 4) & (fxx5 <= 18))[0]
            alpha_power_base[area, trial] = np.max(pxx5[alpha_idx]) if len(alpha_idx) > 0 else 0
    
    print('    Computing lesioned power spectra...')
    for trial in range(ntrials):
        for area in range(Nareas):
            # Lesioned - L2/3 gamma
            fxx2, pxx2 = signal.periodogram(X2_lesion[area, :, trial], fs=fs)
            gamma_idx = np.where((fxx2 >= 30) & (fxx2 <= 70))[0]
            gamma_power_les[area, trial] = np.max(pxx2[gamma_idx]) if len(gamma_idx) > 0 else 0
            
            # Lesioned - L5/6 alpha
            fxx5, pxx5 = signal.periodogram(X5_lesion[area, :, trial], fs=fs)
            alpha_idx = np.where((fxx5 >= 4) & (fxx5 <= 18))[0]
            alpha_power_les[area, trial] = np.max(pxx5[alpha_idx]) if len(alpha_idx) > 0 else 0
    
    # Average across trials
    gamma_power_base_mean = np.mean(gamma_power_base, axis=1)
    alpha_power_base_mean = np.mean(alpha_power_base, axis=1)
    gamma_power_les_mean = np.mean(gamma_power_les, axis=1)
    alpha_power_les_mean = np.mean(alpha_power_les, axis=1)
    
    # Percent change
    gamma_change = 100 * (gamma_power_les_mean - gamma_power_base_mean) / (gamma_power_base_mean + 1e-10)
    alpha_change = 100 * (alpha_power_les_mean - alpha_power_base_mean) / (alpha_power_base_mean + 1e-10)
    
    return {
        'gamma_power_baseline': gamma_power_base_mean,
        'alpha_power_baseline': alpha_power_base_mean,
        'gamma_power_lesion': gamma_power_les_mean,
        'alpha_power_lesion': alpha_power_les_mean,
        'gamma_change': gamma_change,
        'alpha_change': alpha_change
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def largescale_plt(power_results, area_names, roi_indices, output_dir, name='power_by_area'):
    """
    Generate and save all largescale analysis plots.
    
    Parameters
    ----------
    power_results : dict
        Results from largescale_power_analysis
    area_names : list
        Full list of area names
    roi_indices : list or None
        ROI indices
    output_dir : str
        Output directory
    """
    # Create output directory if needed
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    gamma_power = power_results['gamma_power']
    alpha_power = power_results['alpha_power']
    save_path = os.path.join(output_dir, f'{name}.png')

    if roi_indices is not None:
        gamma = gamma_power[roi_indices, :]
        alpha = alpha_power[roi_indices, :]
        names = [area_names[i] for i in roi_indices]
    else:
        gamma = gamma_power
        alpha = alpha_power
        names = area_names
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gamma power
    gamma_mean = np.mean(gamma, axis=1)
    gamma_sem = np.std(gamma, axis=1) / np.sqrt(gamma.shape[1])
    
    axes[0].bar(range(len(names)), gamma_mean, yerr=gamma_sem, 
                capsize=3, color='orange', alpha=0.8)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_xlabel('Cortical Area')
    axes[0].set_ylabel('Gamma Power')
    axes[0].set_title('Gamma (30-70 Hz) Power')
    
    # Alpha power
    alpha_mean = np.mean(alpha, axis=1)
    alpha_sem = np.std(alpha, axis=1) / np.sqrt(alpha.shape[1])
    
    axes[1].bar(range(len(names)), alpha_mean, yerr=alpha_sem,
                capsize=3, color='purple', alpha=0.8)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].set_xlabel('Cortical Area')
    axes[1].set_ylabel('Alpha Power')
    axes[1].set_title('Alpha (6-18 Hz) Power')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    print(f'    Saved: {save_path}')


def lesion_plt(rate_results, power_results, full_area_names, lesion_names, lesion_indices_full, output_dir, roi_indices=None):
    """
    Create visualization of lesion effects on firing rates and oscillatory power.
    Each plot is saved as a separate figure for better readability.
    
    Parameters
    ----------
    rate_results : dict
        Output from lesion_rate_analysis (full network, Nareas=30)
    power_results : dict
        Output from lesion_power_analysis (full network, Nareas=30)
    full_area_names : list of str
        All area names (30 areas)
    lesion_names : list of str
        Names of lesioned areas
    lesion_indices_full : list of int
        Indices of lesioned areas in full network (0-29)
    output_dir : str
        Directory to save plots
    roi_indices : list of int or None
        If provided, only plot these area indices (ROI subset)
    """
    # Determine which areas to plot
    if roi_indices is not None:
        # Use ROI subset
        plot_indices = roi_indices
        area_names = [full_area_names[i] for i in roi_indices]
        # Map lesion indices to ROI space
        lesion_indices_plot = [roi_indices.index(i) for i in lesion_indices_full if i in roi_indices]
        subset_label = ""
    else:
        # Use all areas
        plot_indices = list(range(len(full_area_names)))
        area_names = full_area_names
        lesion_indices_plot = lesion_indices_full
        subset_label = ""
    
    Nareas = len(plot_indices)
    x_pos = np.arange(Nareas)
    lesion_title = f'Lesion: {", ".join(lesion_names)}'
    
    # Extract data for plotted areas
    rate_change = rate_results['percent_change'][plot_indices]
    pvalues = rate_results['pvalues'][plot_indices]
    significant = rate_results['significant'][plot_indices]
    mean_baseline = rate_results['mean_baseline'][plot_indices]
    mean_lesion = rate_results['mean_lesion'][plot_indices]
    gamma_change = power_results['gamma_change'][plot_indices]
    alpha_change = power_results['alpha_change'][plot_indices]
    
    # Create filtered data excluding lesioned areas for figures 1, 3, 4, 5
    non_lesion_mask = [i not in lesion_indices_plot for i in range(Nareas)]
    non_lesion_indices = [i for i in range(Nareas) if i not in lesion_indices_plot]
    
    rate_change_filtered = rate_change[non_lesion_mask]
    pvalues_filtered = pvalues[non_lesion_mask]
    significant_filtered = significant[non_lesion_mask]
    gamma_change_filtered = gamma_change[non_lesion_mask]
    alpha_change_filtered = alpha_change[non_lesion_mask]
    area_names_filtered = [area_names[i] for i in non_lesion_indices]
    
    Nareas_filtered = len(non_lesion_indices)
    x_pos_filtered = np.arange(Nareas_filtered)
    
    # -------------------------------------------------------------------------
    # 1. Firing Rate Changes (excluding lesioned areas)
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    colors = ['steelblue'] * Nareas_filtered
    bars = ax1.bar(x_pos_filtered, rate_change_filtered, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add percentage labels and significance stars on top of bars
    for i, (val, sig) in enumerate(zip(rate_change_filtered, significant_filtered)):
        y_pos = val
        # Position label above or below bar
        if val >= 0:
            va = 'bottom'
            offset = 0.1
        else:
            va = 'top'
            offset = -0.1
        # Add percentage label
        label = f'{val:.1f}%'
        ax1.text(i, y_pos + offset, label, ha='center', va=va, fontsize=8, fontweight='bold' if sig else 'normal')
    
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    ax1.set_xticks(x_pos_filtered)
    ax1.set_xticklabels(area_names_filtered, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('Firing Rate Change (%)', fontsize=12)
    ax1.set_xlabel('Brain Area', fontsize=12)
    ax1.set_title(f'Firing Rate Changes After Lesion{subset_label}\n({lesion_title})', fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.5, Nareas_filtered - 0.5)
    # Expand y-axis to fit labels
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymin - abs(ymin)*0.15, ymax + abs(ymax)*0.15)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'lesion_1_rate_changes.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'    Saved: {save_path}')
    plt.close(fig1)
    
    # -------------------------------------------------------------------------
    # 2. Baseline vs Lesioned Mean Rates
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(x_pos, mean_baseline, 'o-', label='Baseline (Healthy)', 
             linewidth=2, markersize=8, color='dodgerblue')
    ax2.plot(x_pos, mean_lesion, 's-', label='Lesioned', 
             linewidth=2, markersize=8, color='orangered')
    
    # Highlight lesioned areas
    for idx in lesion_indices_plot:
        ax2.axvline(idx, color='darkred', alpha=0.3, linestyle='--', linewidth=2)
        ax2.axvspan(idx - 0.4, idx + 0.4, alpha=0.15, color='darkred')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(area_names, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Mean Firing Rate (Hz)', fontsize=12)
    ax2.set_xlabel('Brain Area', fontsize=12)
    ax2.set_title(f'Absolute Firing Rates: Baseline vs Lesioned{subset_label}\n({lesion_title})', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.set_xlim(-0.5, Nareas - 0.5)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'lesion_2_absolute_rates.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'    Saved: {save_path}')
    plt.close(fig2)
    
    # -------------------------------------------------------------------------
    # 3. Gamma Power Changes (excluding lesioned areas)
    # -------------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    colors = ['orange'] * Nareas_filtered
    ax3.bar(x_pos_filtered, gamma_change_filtered, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add percentage labels on top of bars
    for i, val in enumerate(gamma_change_filtered):
        if val >= 0:
            va = 'bottom'
            offset = 0.1
        else:
            va = 'top'
            offset = -0.1
        ax3.text(i, val + offset, f'{val:.1f}%', ha='center', va=va, fontsize=8)
    
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_xticks(x_pos_filtered)
    ax3.set_xticklabels(area_names_filtered, rotation=45, ha='right', fontsize=10)
    ax3.set_ylabel('Gamma Power Change (%)', fontsize=12)
    ax3.set_xlabel('Brain Area', fontsize=12)
    ax3.set_title(f'Gamma Band Power Changes (30-70 Hz){subset_label}\n({lesion_title})', fontsize=14, fontweight='bold')
    ax3.set_xlim(-0.5, Nareas_filtered - 0.5)
    # Expand y-axis to fit labels
    ymin, ymax = ax3.get_ylim()
    ax3.set_ylim(ymin - abs(ymin)*0.15, ymax + abs(ymax)*0.15)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'lesion_3_gamma_power.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'    Saved: {save_path}')
    plt.close(fig3)
    
    # -------------------------------------------------------------------------
    # 4. Alpha Power Changes (excluding lesioned areas)
    # -------------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    colors = ['purple'] * Nareas_filtered
    ax4.bar(x_pos_filtered, alpha_change_filtered, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add percentage labels on top of bars
    for i, val in enumerate(alpha_change_filtered):
        if val >= 0:
            va = 'bottom'
            offset = 0.1
        else:
            va = 'top'
            offset = -0.1
        ax4.text(i, val + offset, f'{val:.1f}%', ha='center', va=va, fontsize=8)
    
    ax4.axhline(0, color='black', linestyle='-', linewidth=1)
    ax4.set_xticks(x_pos_filtered)
    ax4.set_xticklabels(area_names_filtered, rotation=45, ha='right', fontsize=10)
    ax4.set_ylabel('Alpha Power Change (%)', fontsize=12)
    ax4.set_xlabel('Brain Area', fontsize=12)
    ax4.set_title(f'Alpha Band Power Changes (4-18 Hz){subset_label}\n({lesion_title})', fontsize=14, fontweight='bold')
    ax4.set_xlim(-0.5, Nareas_filtered - 0.5)
    # Expand y-axis to fit labels
    ymin, ymax = ax4.get_ylim()
    ax4.set_ylim(ymin - abs(ymin)*0.15, ymax + abs(ymax)*0.15)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'lesion_4_alpha_power.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'    Saved: {save_path}')
    plt.close(fig4)
    
    # -------------------------------------------------------------------------
    # 5. Distance-Dependent Effects (excluding lesioned areas)
    # -------------------------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    
    # Compute "distance" from lesioned areas (simplified: use index distance)
    # Only compute for non-lesioned areas
    distances_filtered = np.zeros(Nareas_filtered)
    for j, orig_idx in enumerate(non_lesion_indices):
        if len(lesion_indices_plot) > 0:
            # Get original index in full space
            orig_plot_idx = plot_indices[orig_idx]
            # Get original indices of lesioned areas
            lesion_orig_indices = [plot_indices[idx] for idx in lesion_indices_plot]
            # Compute minimum distance
            distances_filtered[j] = min([abs(orig_plot_idx - lesion_orig_idx) for lesion_orig_idx in lesion_orig_indices])
        else:
            distances_filtered[j] = j
    
    # Scatter: distance vs rate change (only non-lesioned areas)
    colors_scatter = ['dodgerblue'] * Nareas_filtered
    sizes = [80] * Nareas_filtered
    ax5.scatter(distances_filtered, rate_change_filtered, c=colors_scatter, s=sizes, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add labels for each point
    for j, name in enumerate(area_names_filtered):
        ax5.annotate(name, (distances_filtered[j], rate_change_filtered[j]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9,
                     alpha=0.8)
    
    ax5.axhline(0, color='black', linestyle='--', linewidth=1)
    ax5.set_xlabel('Index Distance from Lesioned Area', fontsize=12)
    ax5.set_ylabel('Firing Rate Change (%)', fontsize=12)
    ax5.set_title(f'Distance-Dependent Effects of Lesion{subset_label}\n({lesion_title})', fontsize=14, fontweight='bold')
    
    # Add legend (only non-lesioned areas now)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='dodgerblue', markersize=10, label='Non-lesioned Areas')
    ]
    ax5.legend(handles=legend_elements, loc='best', fontsize=10)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'lesion_5_distance_effects.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'    Saved: {save_path}')
    plt.close(fig5)
    
    print(f'    All lesion plots saved to: {output_dir}/')


def modularity_plt(P, Z, partition, area_names, output_dir):
    """
    Create visualization of modularity analysis.
    """
    hub_types = {}

    P_thresh = 0.3  # Participation threshold
    Z_thresh = 1.0  # Hub threshold

    # Visualization Plot
    plt.figure(figsize=(12, 8))
    
    colors = [partition[i] for i in range(len(area_names))]
    scatter = plt.scatter(P, Z, c=colors, cmap='tab20', s=120, alpha=0.7, edgecolors='black')

    extra_space = 0.0075
    for i, txt in enumerate(area_names):
        plt.annotate(txt, (P[i]+extra_space, Z[i]+extra_space), fontsize=9, fontweight='bold')
    
    plt.axvline(x=P_thresh, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    plt.axhline(y=Z_thresh, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Top-Left: Provincial Hubs
    plt.text(0.1, 1.8, 'Provincial Hubs', fontsize=12, color='darkred', ha='center', fontweight='bold')

    # Top-Right: Connector Hubs
    plt.text(0.6, 1.8, 'Connector Hubs', fontsize=12, color='darkgreen', ha='center', fontweight='bold')

    # Bottom-Right: Satellite Connectors
    plt.text(0.6, -1.0, 'Satellite Connectors', fontsize=11, color='teal', ha='center', style='italic')

    # Bottom-Left: Peripheral Nodes
    plt.text(0.1, -0.75, 'Peripheral Nodes', fontsize=11, color='gray', ha='center', style='italic')

    plt.xlabel('P-Coefficient\n← Local Connections ......... Global Connections →', fontsize=11)
    plt.ylabel('Z-Score\n← Low Influence ......... High Influence →', fontsize=11)

    plt.title('Functional Roles of Cortical Areas', fontsize=14)
    plt.grid(True, alpha=0.2)
    
    # Save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hub_classification.png'))
    plt.close()