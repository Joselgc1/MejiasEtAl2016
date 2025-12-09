from __future__ import print_function, division

import os
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import inv, solve, cholesky
from numba import jit
from concurrent.futures import ProcessPoolExecutor
import warnings

from calculate_rate import transduction_function
from helper_functions import calculate_periodogram, matlab_smooth, plt_filled_std, get_network_configuration

# Set random seed for reproducibility
np.random.seed(42)

# Check for optional dependencies
HAS_NITIME = False
try:
    from nitime.analysis import GrangerAnalyzer
    from nitime import TimeSeries
    HAS_NITIME = True
except ImportError:
    pass


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
    roi_names = ['V1', 'V2', 'V4', 'DP', '8m', '8l', 'TEO', '7A']
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
    
    # Selectivity matrix for FB projections (proportion targeting different layers)
    s = np.ascontiguousarray(0.1 * np.ones((Nareas, Nareas)))
    
    return Wff, Wfb, np.ascontiguousarray(delays), s


# ============================================================================
# SIMULATION
# ============================================================================

def largescale_simulation(par, Wff, Wfb, delays, s, Iext, triallength=200, transient=20, seed=None):
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
    triallength : float
        Simulation length in seconds
    transient : float
        Transient period to discard in seconds
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
    Nt = int(round(triallength / dt))
    
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
    # This corresponds to MATLAB: if time>0.2 (little transient >59ms)
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
        
        # Update rates (vectorized across populations)
        rate[:, t, :] = irate + tstep_vec * (-irate + transfer) + tstep2_vec * xi[:, t, :]
    
    return rate


def _simulate_trial_worker(args):
    """
    Helper for parallel trial execution (must be top-level for pickling).
    """
    (trial_idx, par, Wff, Wfb, delays, s, Iext, triallength, transient,
     nobs_binned, t0_idx, binx, eta, base_seed) = args

    seed = None if base_seed is None else base_seed + trial_idx
    rate_local = largescale_simulation(par, Wff, Wfb, delays, s, Iext,
                                       triallength, transient, seed=seed)
    Nareas = par['Nareas']
    X_loc = np.zeros((Nareas, nobs_binned))
    X2_loc = np.zeros((Nareas, nobs_binned))
    X5_loc = np.zeros((Nareas, nobs_binned))

    for area in range(Nareas):
        r2 = rate_local[0, t0_idx::binx, area][:nobs_binned]
        r5 = rate_local[2, t0_idx::binx, area][:nobs_binned]
        X2_loc[area, :] = r2
        X5_loc[area, :] = r5
        X_loc[area, :] = eta * r2 + (1 - eta) * r5

    return trial_idx, X_loc, X2_loc, X5_loc


def run_largescale_trials(par, Wff, Wfb, delays, s, Iext, 
                          triallength=200, transient=20, ntrials=30,
                          use_parallel=False, max_workers=None, base_seed=42):
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
    triallength : float
        Trial length in seconds
    transient : float
        Transient to discard
    ntrials : int
        Number of trials
    use_parallel : bool
        If True, distribute trials across processes
    max_workers : int or None
        Workers for the process pool (None -> os.cpu_count())
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
    nobs_full = int(round((triallength - transient) / dt))
    nobs_binned = nobs_full // binx
    t0_idx = int(round(transient / dt))
    
    # Initialize arrays
    X = np.zeros((Nareas, nobs_binned, ntrials))
    X2 = np.zeros((Nareas, nobs_binned, ntrials))
    X5 = np.zeros((Nareas, nobs_binned, ntrials))

    if use_parallel and ntrials > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            args_iter = (
                (idx, par, Wff, Wfb, delays, s, Iext, triallength, transient,
                 nobs_binned, t0_idx, binx, eta, base_seed)
                for idx in range(ntrials)
            )
            for trial_idx, X_loc, X2_loc, X5_loc in ex.map(_simulate_trial_worker, args_iter):
                X[:, :, trial_idx] = X_loc
                X2[:, :, trial_idx] = X2_loc
                X5[:, :, trial_idx] = X5_loc
    else:
        for trial in range(ntrials):
            print(f'    Running trial {trial+1} of {ntrials}...')
            _, X_loc, X2_loc, X5_loc = _simulate_trial_worker(
                (trial, par, Wff, Wfb, delays, s, Iext, triallength, transient,
                 nobs_binned, t0_idx, binx, eta, base_seed)
            )
            X[:, :, trial] = X_loc
            X2[:, :, trial] = X2_loc
            X5[:, :, trial] = X5_loc
    
    return X, X2, X5


# ============================================================================
# GRANGER CAUSALITY ANALYSIS
# ============================================================================

def estimate_var_order(X, max_order=30, criterion='BIC'):
    """
    Estimate optimal VAR model order using information criteria.
    
    Parameters
    ----------
    X : ndarray (Nareas, T)
        Time series data (concatenated trials)
    max_order : int
        Maximum model order to test
    criterion : str
        'AIC' or 'BIC'
        
    Returns
    -------
    best_order : int
        Optimal model order
    """
    Nareas, T = X.shape
    
    ic_values = []
    
    for order in range(1, min(max_order + 1, T // (2 * Nareas))):
        try:
            # Build lagged matrices
            Y = X[:, order:]
            n_obs = T - order
            
            X_lag = np.zeros((Nareas * order, n_obs))
            for lag in range(order):
                X_lag[lag*Nareas:(lag+1)*Nareas, :] = X[:, order-lag-1:T-lag-1]
            
            # OLS estimation
            XXT = X_lag @ X_lag.T + 1e-8 * np.eye(X_lag.shape[0])
            A = Y @ X_lag.T @ inv(XXT)
            
            # Residuals and covariance
            E = Y - A @ X_lag
            Sigma = E @ E.T / n_obs
            
            # Log determinant of covariance
            log_det = np.log(np.linalg.det(Sigma) + 1e-10)
            
            # Information criteria
            n_params = Nareas * Nareas * order
            if criterion == 'AIC':
                ic = log_det + 2 * n_params / n_obs
            else:  # BIC
                ic = log_det + np.log(n_obs) * n_params / n_obs
            
            ic_values.append((order, ic))
        except:
            continue
    
    if len(ic_values) == 0:
        return 10  # Default
    
    best_order = min(ic_values, key=lambda x: x[1])[0]
    return best_order


def fit_var_model(X, order):
    """
    Fit a VAR model using OLS.
    
    Parameters
    ----------
    X : ndarray (Nareas, T)
        Time series data
    order : int
        Model order
        
    Returns
    -------
    A : ndarray (Nareas, Nareas * order)
        VAR coefficients
    Sigma : ndarray (Nareas, Nareas)
        Residual covariance matrix
    """
    Nareas, T = X.shape
    
    # Subtract mean
    X = X - X.mean(axis=1, keepdims=True)
    
    # Build lagged matrices
    Y = X[:, order:]
    n_obs = T - order
    
    X_lag = np.zeros((Nareas * order, n_obs))
    for lag in range(order):
        X_lag[lag*Nareas:(lag+1)*Nareas, :] = X[:, order-lag-1:T-lag-1]
    
    # OLS estimation: A = Y @ X_lag.T @ inv(X_lag @ X_lag.T)
    XXT = X_lag @ X_lag.T + 1e-8 * np.eye(X_lag.shape[0])
    A = Y @ X_lag.T @ inv(XXT)
    
    # Residual covariance
    E = Y - A @ X_lag
    Sigma = E @ E.T / n_obs
    
    return A, Sigma


def var_to_autocov(A, Sigma, max_lags=None):
    """
    Compute autocovariance sequence from VAR parameters.
    
    Parameters
    ----------
    A : ndarray (Nareas, Nareas * order)
        VAR coefficients
    Sigma : ndarray (Nareas, Nareas)
        Residual covariance
    max_lags : int
        Maximum lags to compute (default: automatic)
        
    Returns
    -------
    G : ndarray (Nareas, Nareas, max_lags)
        Autocovariance sequence
    """
    Nareas = Sigma.shape[0]
    order = A.shape[1] // Nareas
    
    if max_lags is None:
        max_lags = min(1000, 10 * order)
    
    # Construct companion matrix
    A_comp = np.zeros((Nareas * order, Nareas * order))
    A_comp[:Nareas, :] = A
    if order > 1:
        A_comp[Nareas:, :-Nareas] = np.eye(Nareas * (order - 1))
    
    # Initialize autocovariance
    G = np.zeros((Nareas, Nareas, max_lags))
    
    # Solve Lyapunov equation for lag-0 autocovariance
    # G0 = A_comp @ G0 @ A_comp.T + [Sigma, 0; 0, 0]
    Sigma_comp = np.zeros((Nareas * order, Nareas * order))
    Sigma_comp[:Nareas, :Nareas] = Sigma
    
    # Iterative solution (stable for stationary processes)
    G0_comp = Sigma_comp.copy()
    for _ in range(100):
        G0_comp_new = A_comp @ G0_comp @ A_comp.T + Sigma_comp
        if np.max(np.abs(G0_comp_new - G0_comp)) < 1e-10:
            break
        G0_comp = G0_comp_new
    
    G[:, :, 0] = G0_comp[:Nareas, :Nareas]
    
    # Compute higher lags recursively
    G_curr = G0_comp
    for lag in range(1, max_lags):
        G_curr = A_comp @ G_curr
        G[:, :, lag] = G_curr[:Nareas, :Nareas]
    
    return G


def autocov_to_cpsd(G, fs, fres):
    """
    Compute cross-power spectral density from autocovariance.
    
    Parameters
    ----------
    G : ndarray (Nareas, Nareas, max_lags)
        Autocovariance sequence
    fs : float
        Sampling frequency
    fres : int
        Frequency resolution
        
    Returns
    -------
    S : ndarray (Nareas, Nareas, fres)
        Cross-power spectral density
    freqs : ndarray (fres,)
        Frequency vector
    """
    Nareas = G.shape[0]
    max_lags = G.shape[2]
    
    freqs = np.linspace(0, fs/2, fres)
    S = np.zeros((Nareas, Nareas, fres), dtype=complex)
    
    for fi, f in enumerate(freqs):
        S_f = G[:, :, 0].copy().astype(complex)
        for lag in range(1, max_lags):
            z = np.exp(-2j * np.pi * f * lag / fs)
            S_f += G[:, :, lag] * z + G[:, :, lag].T * np.conj(z)
        S[:, :, fi] = S_f / fs
    
    return S, freqs


def compute_spectral_granger_var(X, fs, order=None, fres=100, max_order=30):
    """
    Compute spectral Granger causality using proper VAR modeling.
    
    This implements the Geweke (1982) spectral decomposition method,
    matching the MVGC toolbox approach.
    
    Parameters
    ----------
    X : ndarray (Nareas, Nt, ntrials)
        Time series data
    fs : float
        Sampling frequency
    order : int or None
        VAR model order (if None, estimated using BIC)
    fres : int
        Frequency resolution
    max_order : int
        Maximum order for BIC estimation
        
    Returns
    -------
    f_gc : ndarray (Nareas, Nareas, fres)
        Spectral Granger causality
    freqs : ndarray (fres,)
        Frequency vector
    """
    Nareas, Nt, ntrials = X.shape
    
    # Concatenate trials
    X_all = X.reshape(Nareas, -1)
    X_all = X_all - X_all.mean(axis=1, keepdims=True)
    
    # Estimate model order if not provided
    if order is None:
        order = estimate_var_order(X_all, max_order=max_order, criterion='BIC')
        print(f'    Estimated VAR order (BIC): {order}')
    
    # Fit full VAR model
    A, Sigma = fit_var_model(X_all, order)
    
    # Compute autocovariance
    G = var_to_autocov(A, Sigma)
    
    # Compute spectral density
    S, freqs = autocov_to_cpsd(G, fs, fres)
    
    # Compute pairwise conditional Granger causality
    f_gc = np.zeros((Nareas, Nareas, fres))
    
    for j in range(Nareas):  # Source
        for i in range(Nareas):  # Target
            if i != j:
                # For GC from j to i, we need:
                # 1. Full model: predict i using all variables
                # 2. Reduced model: predict i without j
                
                # Get indices for reduced model (all except j)
                reduced_idx = [k for k in range(Nareas) if k != j]
                
                # Fit reduced VAR model
                X_reduced = X_all[reduced_idx, :]
                try:
                    A_red, Sigma_red = fit_var_model(X_reduced, order)
                    G_red = var_to_autocov(A_red, Sigma_red)
                    S_red, _ = autocov_to_cpsd(G_red, fs, fres)
                    
                    # Index of i in reduced model
                    i_red = reduced_idx.index(i) if i in reduced_idx else 0
                    
                    # Spectral GC: log(S_ii_reduced / S_ii_full)
                    for fi in range(fres):
                        S_ii_full = np.real(S[i, i, fi])
                        S_ii_red = np.real(S_red[i_red, i_red, fi])
                        
                        if S_ii_full > 1e-10 and S_ii_red > 1e-10:
                            f_gc[i, j, fi] = np.log(S_ii_red / S_ii_full)
                        else:
                            f_gc[i, j, fi] = 0
                except:
                    # If reduced model fails, use zero
                    f_gc[i, j, :] = 0
    
    # Ensure non-negative (GC is always >= 0)
    f_gc = np.maximum(f_gc, 0)
    
    return f_gc, freqs


def compute_spectral_granger(X, fs, fmax=100, nperseg=None, method='var', order=None):
    """
    Compute spectral Granger causality between all pairs of areas.
    
    Parameters
    ----------
    X : ndarray (Nareas, Nt, ntrials)
        Time series data
    fs : float
        Sampling frequency
    fmax : float
        Maximum frequency to compute
    nperseg : int
        Segment length for spectral estimation (coherence method only)
    method : str
        'var' for VAR-based (recommended), 'coherence' for simplified approach
    order : int or None
        VAR model order (if None, estimated using BIC)
        
    Returns
    -------
    f_gc : ndarray (Nareas, Nareas, Nfreq)
        Spectral Granger causality estimates
    freqs : ndarray (Nfreq,)
        Frequency vector
    """
    # Use VAR-based method by default (matches MATLAB MVGC)
    if method == 'var':
        fres = int(fmax * 2)  # Frequency resolution
        f_gc, freqs = compute_spectral_granger_var(X, fs, order=order, fres=fres)
        
        # Trim to fmax
        freq_mask = freqs <= fmax
        return f_gc[:, :, freq_mask], freqs[freq_mask]
    
    # Fallback: coherence-based approximation
    Nareas, Nt, ntrials = X.shape
    
    if nperseg is None:
        nperseg = min(256, Nt // 4)
    
    # Concatenate trials for spectral estimation
    X_concat = X.reshape(Nareas, -1)
    
    # Compute cross-spectral density for all pairs
    freqs = None
    Sxy = None
    
    for i in range(Nareas):
        for j in range(Nareas):
            f, Pxy = signal.csd(X_concat[i, :], X_concat[j, :], 
                               fs=fs, nperseg=nperseg, noverlap=nperseg//2)
            if freqs is None:
                freqs = f
                Nfreq = len(f[f <= fmax])
                Sxy = np.zeros((Nareas, Nareas, Nfreq), dtype=complex)
            
            Sxy[i, j, :] = Pxy[:Nfreq]
    
    freqs = freqs[:Nfreq]
    
    # Compute auto-spectra
    Sxx = np.zeros((Nareas, Nfreq))
    for i in range(Nareas):
        f, Pxx = signal.welch(X_concat[i, :], fs=fs, 
                             nperseg=nperseg, noverlap=nperseg//2)
        Sxx[i, :] = Pxx[:Nfreq]
    
    # Compute spectral Granger causality using coherence approximation
    # Note: This is a simplified approach, VAR method is more accurate
    f_gc = np.zeros((Nareas, Nareas, Nfreq))
    
    for i in range(Nareas):
        for j in range(Nareas):
            if i != j:
                # Squared coherence
                coh = np.abs(Sxy[i, j, :]) ** 2 / (Sxx[i, :] * Sxx[j, :] + 1e-10)
                # Transform to GC-like measure
                f_gc[i, j, :] = -np.log(1 - np.clip(coh, 0, 0.999))
    
    return f_gc, freqs


def compute_granger_nitime(X, fs, order=30, fmax=100):
    """
    Compute Granger causality using nitime package (if available).
    
    This provides the most accurate MVGC implementation.
    
    Parameters
    ----------
    X : ndarray (Nareas, Nt, ntrials)
        Time series data
    fs : float
        Sampling frequency
    order : int
        VAR model order
    fmax : float
        Maximum frequency
        
    Returns
    -------
    f_gc : ndarray (Nareas, Nareas, Nfreq)
        Spectral Granger causality
    freqs : ndarray (Nfreq,)
        Frequency vector
    """
    if not HAS_NITIME:
        warnings.warn("nitime not available, falling back to VAR method")
        return compute_spectral_granger(X, fs, fmax=fmax, method='var')
    
    Nareas, Nt, ntrials = X.shape
    X_concat = X.reshape(Nareas, -1)
    
    ts = TimeSeries(X_concat, sampling_rate=fs)
    ga = GrangerAnalyzer(ts, order=order)
    
    freqs = ga.frequencies
    freq_mask = freqs <= fmax
    
    # Get spectral Granger causality (causality_xy[i,j] = GC from j to i)
    f_gc = ga.causality_xy[:, :, freq_mask]
    
    return f_gc, freqs[freq_mask]


# ============================================================================
# DAI AND MDAI CALCULATIONS
# ============================================================================

def compute_dai(f_gc):
    """
    Compute Directed Asymmetry Index (DAI) from spectral Granger causality.
    
    DAI(i,j,f) = (GC(i->j,f) - GC(j->i,f)) / (GC(i->j,f) + GC(j->i,f))
    
    Parameters
    ----------
    f_gc : ndarray (Nareas, Nareas, Nfreq)
        Spectral Granger causality
        
    Returns
    -------
    dai : ndarray (Nareas, Nareas, Nfreq)
        Directed asymmetry index
    """
    Nareas = f_gc.shape[0]
    Nfreq = f_gc.shape[2]
    
    dai = np.zeros((Nareas, Nareas, Nfreq))
    
    for i in range(Nareas):
        for j in range(Nareas):
            if i != j:
                gc_ij = f_gc[i, j, :]  # j -> i
                gc_ji = f_gc[j, i, :]  # i -> j
                denom = gc_ij + gc_ji + 1e-10
                dai[i, j, :] = (gc_ij - gc_ji) / denom
    
    return dai


def compute_mdai(dai, freqs, alpha_range=(6, 18), gamma_range=(30, 70)):
    """
    Compute multifrequency DAI (mDAI).
    
    mDAI = (DAI_gamma - DAI_alpha) / 2
    
    Parameters
    ----------
    dai : ndarray (Nareas, Nareas, Nfreq)
        Directed asymmetry index
    freqs : ndarray
        Frequency vector
    alpha_range : tuple
        Alpha frequency range (Hz)
    gamma_range : tuple
        Gamma frequency range (Hz)
        
    Returns
    -------
    mdai : ndarray (Nareas, Nareas)
        Multifrequency DAI
    """
    # Find frequency indices
    alpha_idx = np.where((freqs >= alpha_range[0]) & (freqs <= alpha_range[1]))[0]
    gamma_idx = np.where((freqs >= gamma_range[0]) & (freqs <= gamma_range[1]))[0]
    
    # Average DAI in each frequency band
    dai_alpha = np.mean(dai[:, :, alpha_idx], axis=2)
    dai_gamma = np.mean(dai[:, :, gamma_idx], axis=2)
    
    # mDAI calculation (note: negative sign for alpha due to negative correlation)
    mdai = (dai_gamma - dai_alpha) / 2
    
    # Set diagonal to zero
    np.fill_diagonal(mdai, 0)
    
    return mdai


def compute_sln_dai_correlation(dai, sln_mat, freqs, roi_indices=None):
    """
    Compute correlation between SLN and DAI at each frequency.
    
    Parameters
    ----------
    dai : ndarray (Nareas, Nareas, Nfreq)
        Directed asymmetry index
    sln_mat : ndarray (Nareas, Nareas)
        SLN matrix
    freqs : ndarray
        Frequency vector
    roi_indices : list
        Indices of ROIs to use (if None, use all)
        
    Returns
    -------
    sln_dai_corr : ndarray (Nfreq,)
        Spearman correlation at each frequency
    """
    if roi_indices is not None:
        sln = sln_mat[np.ix_(roi_indices, roi_indices)]
        dai_roi = dai[np.ix_(roi_indices, roi_indices, range(dai.shape[2]))]
    else:
        sln = sln_mat
        dai_roi = dai
    
    Nareas = sln.shape[0]
    Nfreq = dai_roi.shape[2]
    
    # Flatten to vectors (excluding diagonal and zero entries)
    mask = (sln > 0) & (sln < 1)  # Exclude pure FF/FB and self-connections
    sln_vec = sln[mask]
    
    sln_dai_corr = np.zeros(Nfreq)
    
    for fi in range(Nfreq):
        dai_fi = dai_roi[:, :, fi]
        dai_vec = dai_fi[mask]
        
        if len(sln_vec) > 2:
            rho, _ = spearmanr(sln_vec, dai_vec)
            sln_dai_corr[fi] = rho
    
    return sln_dai_corr


def compute_mdai_sln_correlation(mdai, sln_mat, roi_indices=None):
    """
    Compute correlation between mDAI and SLN.
    
    Parameters
    ----------
    mdai : ndarray (Nareas, Nareas)
        Multifrequency DAI
    sln_mat : ndarray (Nareas, Nareas)
        SLN matrix
    roi_indices : list
        Indices of ROIs to use
        
    Returns
    -------
    rho : float
        Spearman correlation coefficient
    pval : float
        P-value
    """
    if roi_indices is not None:
        sln = sln_mat[np.ix_(roi_indices, roi_indices)]
        mdai_roi = mdai[np.ix_(roi_indices, roi_indices)]
    else:
        sln = sln_mat
        mdai_roi = mdai
    
    # Flatten to vectors (excluding diagonal and zero entries)
    mask = (sln > 0) & (sln < 1)
    sln_vec = sln[mask]
    mdai_vec = mdai_roi[mask]
    
    rho, pval = spearmanr(sln_vec, mdai_vec)
    
    return rho, pval, sln_vec, mdai_vec


# ============================================================================
# FUNCTIONAL HIERARCHY
# ============================================================================

def compute_functional_hierarchy(mdai, adj=None):
    """
    Compute functional hierarchical positions from mDAI values.
    
    Based on Bastos et al. (2015) method and MATLAB ranks2.m implementation.
    
    The algorithm:
    1. Rescale mDAI to range approximately -5 to 5
    2. For each column j (source area), find the minimum value
    3. Shift all values in column j so the minimum becomes 1
    4. Apply adjacency mask to only consider connected pairs
    5. For each area i, compute mean of row i values (incoming connections)
    
    Parameters
    ----------
    mdai : ndarray (Nareas, Nareas)
        Multifrequency DAI matrix
    adj : ndarray (Nareas, Nareas)
        Adjacency matrix (if None, use non-zero mdai entries)
        
    Returns
    -------
    hierarchy : ndarray (Nareas,)
        Hierarchical position of each area
    hierarchy_sem : ndarray (Nareas,)
        Standard error of the mean
    """
    Nareas = mdai.shape[0]
    
    # Rescale mDAI: multiply by 5 (as in MATLAB ranks2.m)
    mdai_scaled = 5.0 * mdai
    
    # For each column (source), find minimum and shift so min = 1
    # This matches MATLAB: z1(:,i)=min(mDAIp(:,i)); mDAIp=mDAIp+ones-z1;
    min_per_col = np.min(mdai_scaled, axis=0, keepdims=True)  # Shape: (1, Nareas)
    # Broadcast: subtract min from each column and add 1
    mdai_shifted = mdai_scaled - min_per_col + 1.0
    
    # Adjacency matrix: identifies functionally connected pairs
    if adj is None:
        # Use non-zero absolute mDAI values as adjacency
        adj = (np.abs(mdai) > 0.01).astype(float)
    
    # Apply adjacency mask
    mdai_adj = mdai_shifted * adj
    
    # Compute hierarchical position for each area
    # For area i, consider row i (incoming connections from all sources)
    hierarchy = np.zeros(Nareas)
    hierarchy_sem = np.zeros(Nareas)
    
    for i in range(Nareas):
        # Get all non-zero values in row i (functionally connected pairs)
        # This matches MATLAB: [~,~,z0]=find(a0(i,:))
        row_vals = mdai_adj[i, :]
        connected_vals = row_vals[row_vals > 0]
        
        if len(connected_vals) > 0:
            hierarchy[i] = np.mean(connected_vals)
            # SEM calculation (as in MATLAB SEM_calc)
            hierarchy_sem[i] = np.std(connected_vals, ddof=1) / np.sqrt(len(connected_vals))
        else:
            hierarchy[i] = 0
            hierarchy_sem[i] = 0
    
    return hierarchy, hierarchy_sem


def compute_anatomical_hierarchy(sln_mat, fln_mat=None):
    """
    Compute anatomical hierarchy from SLN values.
    
    SLN > 0.5 indicates feedforward connection
    SLN < 0.5 indicates feedback connection
    
    Parameters
    ----------
    sln_mat : ndarray (Nareas, Nareas)
        SLN matrix
    fln_mat : ndarray (Nareas, Nareas), optional
        FLN matrix for weighting
        
    Returns
    -------
    hierarchy : ndarray (Nareas,)
        Anatomical hierarchical position
    """
    Nareas = sln_mat.shape[0]
    
    if fln_mat is not None:
        # Weight by connection strength
        weights = fln_mat
    else:
        weights = np.ones_like(sln_mat)
    
    # Mask for valid connections (non-zero, non-diagonal)
    mask = (sln_mat > 0) & (sln_mat < 1)
    np.fill_diagonal(mask, False)
    
    hierarchy = np.zeros(Nareas)
    
    for i in range(Nareas):
        # Average SLN of incoming connections
        incoming_mask = mask[:, i]  # Connections TO area i
        if np.any(incoming_mask):
            weighted_sln = np.sum(sln_mat[incoming_mask, i] * weights[incoming_mask, i])
            total_weight = np.sum(weights[incoming_mask, i])
            hierarchy[i] = weighted_sln / total_weight if total_weight > 0 else 0.5
    
    return hierarchy


# ============================================================================
# POWER ANALYSIS
# ============================================================================

def compute_power_spectra(X2, X5, par, transient=0):
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
    
    return gamma_power, alpha_power


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_sln_dai_correlation(freqs, sln_dai_corr, save_path=None):
    """
    Plot SLN-DAI correlation vs frequency (Figure 6E).
    """
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, sln_dai_corr, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Shade alpha and gamma ranges
    plt.axvspan(6, 18, alpha=0.2, color='red', label='Alpha')
    plt.axvspan(30, 70, alpha=0.2, color='blue', label='Gamma')
    
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('SLN-DAI Correlation (Spearman)', fontsize=12)
    plt.title('Correlation between Anatomical (SLN) and Functional (DAI) Connectivity', fontsize=12)
    plt.legend()
    plt.xlim([0, 100])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return plt.gcf()


def plot_mdai_sln_scatter(sln_vec, mdai_vec, rho, pval, save_path=None):
    """
    Plot mDAI vs SLN scatter (Figure 6F).
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(sln_vec, mdai_vec, alpha=0.5, s=30)
    
    # Fit line
    z = np.polyfit(sln_vec, mdai_vec, 1)
    p = np.poly1d(z)
    x_line = np.linspace(np.min(sln_vec), np.max(sln_vec), 100)
    plt.plot(x_line, p(x_line), 'r-', linewidth=2)
    
    plt.xlabel('SLN (Anatomical)', fontsize=12)
    plt.ylabel('mDAI (Functional)', fontsize=12)
    plt.title(f'SLN vs mDAI Correlation\nr = {rho:.3f}, p = {pval:.2e}', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return plt.gcf()


def plot_functional_hierarchy(hierarchy, hierarchy_sem, area_names, save_path=None):
    """
    Plot functional hierarchy (Figure 6G).
    """
    # Sort by hierarchy
    sort_idx = np.argsort(hierarchy)
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(hierarchy))
    plt.bar(x, hierarchy[sort_idx], yerr=hierarchy_sem[sort_idx], 
            capsize=3, color='steelblue', alpha=0.8)
    
    plt.xticks(x, [area_names[i] for i in sort_idx], rotation=45, ha='right')
    plt.xlabel('Cortical Area', fontsize=12)
    plt.ylabel('Functional Hierarchy', fontsize=12)
    plt.title('Functional Hierarchy of Cortical Areas', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return plt.gcf()


def plot_power_by_area(gamma_power, alpha_power, area_names, roi_indices=None, 
                       save_path=None):
    """
    Plot alpha and gamma power by area (Figure 6C-D style).
    """
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
    
    return fig


def plot_granger_matrix(f_gc, freqs, area_names, roi_indices=None,
                        freq_band='gamma', save_path=None):
    """
    Plot Granger causality matrix for a specific frequency band.
    """
    if freq_band == 'gamma':
        freq_idx = np.where((freqs >= 30) & (freqs <= 70))[0]
        title = 'Granger Causality (Gamma 30-70 Hz)'
    elif freq_band == 'alpha':
        freq_idx = np.where((freqs >= 6) & (freqs <= 18))[0]
        title = 'Granger Causality (Alpha 6-18 Hz)'
    else:
        freq_idx = np.arange(len(freqs))
        title = 'Granger Causality (All frequencies)'
    
    gc_band = np.mean(f_gc[:, :, freq_idx], axis=2)
    
    if roi_indices is not None:
        gc_band = gc_band[np.ix_(roi_indices, roi_indices)]
        names = [area_names[i] for i in roi_indices]
    else:
        names = area_names
    
    plt.figure(figsize=(10, 8))
    plt.imshow(gc_band, cmap='hot', aspect='auto')
    plt.colorbar(label='Granger Causality')
    
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.yticks(range(len(names)), names)
    plt.xlabel('Source Area')
    plt.ylabel('Target Area')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return plt.gcf()


# ============================================================================
# MODULAR ANALYSIS FUNCTIONS (similar to intralaminar/interlaminar)
# ============================================================================

def largescale_granger_analysis(X, par, roi_indices=None, gc_method='var', gc_order=None):
    """
    Compute Granger causality analysis on simulation data.
    
    Parameters
    ----------
    X : ndarray (Nareas, Nt_binned, ntrials)
        Combined L2/3+L5/6 activity
    par : dict
        Parameter dictionary
    roi_indices : list or None
        Indices of ROIs to use (if None, use all areas)
    gc_method : str
        'var' (recommended), 'coherence', or 'nitime'
    gc_order : int or None
        VAR model order (if None, estimated using BIC)
        
    Returns
    -------
    results : dict
        Dictionary with f_gc, freqs, dai, mdai
    """
    # Sampling frequency for Granger analysis
    fs = 1 / (par['dt'] * par['binx'])
    
    # Select ROI subset if specified
    if roi_indices is not None:
        X_analysis = X[roi_indices, :, :]
    else:
        X_analysis = X
    
    # Compute Granger causality
    if gc_method == 'nitime' and HAS_NITIME:
        f_gc, freqs = compute_granger_nitime(X_analysis, fs, order=gc_order or 30)
    else:
        f_gc, freqs = compute_spectral_granger(X_analysis, fs, method=gc_method, order=gc_order)
    
    # Compute DAI and mDAI
    dai = compute_dai(f_gc)
    mdai = compute_mdai(dai, freqs)
    
    return {
        'f_gc': f_gc,
        'freqs': freqs,
        'dai': dai,
        'mdai': mdai,
        'gc_method': gc_method,
        'gc_order': gc_order
    }


def largescale_hierarchy_analysis(f_gc, freqs, sln_mat, roi_indices=None):
    """
    Compute hierarchy and SLN-DAI correlations.
    
    Parameters
    ----------
    f_gc : ndarray (Nareas, Nareas, Nfreq)
        Spectral Granger causality
    freqs : ndarray
        Frequency vector
    sln_mat : ndarray
        SLN matrix (full)
    roi_indices : list or None
        Indices of ROIs used in GC analysis
        
    Returns
    -------
    results : dict
        Dictionary with hierarchy, correlations, etc.
    """
    # Compute DAI and mDAI
    dai = compute_dai(f_gc)
    mdai = compute_mdai(dai, freqs)
    
    # Get SLN subset if using ROIs
    if roi_indices is not None:
        sln_subset = sln_mat[np.ix_(roi_indices, roi_indices)]
    else:
        sln_subset = sln_mat
    
    # Compute SLN-DAI correlation at each frequency
    sln_dai_corr = compute_sln_dai_correlation(dai, sln_subset, freqs)
    
    # Compute mDAI-SLN correlation
    rho, pval, sln_vec, mdai_vec = compute_mdai_sln_correlation(mdai, sln_subset)
    
    # Compute functional hierarchy
    hierarchy_func, hierarchy_sem = compute_functional_hierarchy(mdai)
    
    return {
        'dai': dai,
        'mdai': mdai,
        'sln_dai_correlation': sln_dai_corr,
        'mdai_sln_rho': rho,
        'mdai_sln_pval': pval,
        'sln_vec': sln_vec,
        'mdai_vec': mdai_vec,
        'hierarchy_functional': hierarchy_func,
        'hierarchy_sem': hierarchy_sem
    }


def largescale_power_analysis(X2, X5, par):
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
        
    Returns
    -------
    results : dict
        Dictionary with gamma_power, alpha_power
    """
    gamma_power, alpha_power = compute_power_spectra(X2, X5, par)
    
    return {
        'gamma_power': gamma_power,
        'alpha_power': alpha_power
    }


def largescale_plt(gc_results, hierarchy_results, power_results,
                   area_names, plot_names, roi_indices, output_dir):
    """
    Generate and save all largescale analysis plots.
    
    Parameters
    ----------
    gc_results : dict
        Results from largescale_granger_analysis
    hierarchy_results : dict
        Results from largescale_hierarchy_analysis
    power_results : dict
        Results from largescale_power_analysis
    area_names : list
        Full list of area names
    plot_names : list
        Names to use for plotting (ROI names or full)
    roi_indices : list or None
        ROI indices
    output_dir : str
        Output directory
    """
    # Create output directory if needed
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    freqs = gc_results['freqs']
    f_gc = gc_results['f_gc']
    
    # Plot SLN-DAI correlation (Figure 6E)
    plot_sln_dai_correlation(
        freqs, hierarchy_results['sln_dai_correlation'],
        os.path.join(output_dir, 'sln_dai_correlation.png')
    )
    print(f'    Saved: {output_dir}/sln_dai_correlation.png')
    
    # Plot mDAI-SLN scatter (Figure 6F)
    plot_mdai_sln_scatter(
        hierarchy_results['sln_vec'], hierarchy_results['mdai_vec'],
        hierarchy_results['mdai_sln_rho'], hierarchy_results['mdai_sln_pval'],
        os.path.join(output_dir, 'mdai_sln_scatter.png')
    )
    print(f'    Saved: {output_dir}/mdai_sln_scatter.png')
    
    # Plot functional hierarchy (Figure 6G)
    plot_functional_hierarchy(
        hierarchy_results['hierarchy_functional'],
        hierarchy_results['hierarchy_sem'],
        plot_names,
        os.path.join(output_dir, 'functional_hierarchy.png')
    )
    print(f'    Saved: {output_dir}/functional_hierarchy.png')
    
    # Plot power by area
    plot_power_by_area(
        power_results['gamma_power'], power_results['alpha_power'],
        area_names, roi_indices,
        os.path.join(output_dir, 'power_by_area.png')
    )
    print(f'    Saved: {output_dir}/power_by_area.png')
    
    # Plot Granger matrices
    plot_granger_matrix(f_gc, freqs, plot_names, None, 'gamma',
                        os.path.join(output_dir, 'granger_gamma.png'))
    print(f'    Saved: {output_dir}/granger_gamma.png')
    
    plot_granger_matrix(f_gc, freqs, plot_names, None, 'alpha',
                        os.path.join(output_dir, 'granger_alpha.png'))
    print(f'    Saved: {output_dir}/granger_alpha.png')
