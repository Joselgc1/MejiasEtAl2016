import numpy as np
from numpy.random import normal
from scipy.stats import ttest_ind
import matplotlib.pylab as plt

from calculate_rate import calculate_rate
from helper_functions import calculate_periodogram, find_peak_frequency, matlab_smooth, plt_filled_std


def build_interareal_W(Nareas):
    # Interareal connectivity matrix
    # areas: 0 = V1, 1 = V4
    # pops:  0 = L2E, 1 = L2I, 2 = L5E, 3 = L5I

    J_FF1 = 1.0
    J_FB1 = 0.1
    J_FB2 = 0.5
    J_FB3 = 0.9
    J_FB4 = 0.5

    W = np.zeros((Nareas, Nareas, 4, 4))

    # Feedforward: V1 (area 0) → V4 (area 1)
    # from V1 L2/3E to V4 L2/3E
    W[1, 0, 0, 0] = J_FF1

    # Feedback: V4 (area 1) → V1 (area 0)
    # from V4 L5/6E to V1 L2/3E
    W[0, 1, 0, 2] = J_FB1   # V4 L5/6E → V1 L2/3E
    W[0, 1, 1, 2] = J_FB2   # V4 L5/6E → V1 L2/3I
    W[0, 1, 2, 2] = J_FB3   # V4 L5/6E → V1 L5/6E
    W[0, 1, 3, 2] = J_FB4   # V4 L5/6I → V1 L5/6E

    return W


def interareal_simulation(
    t, dt, tstop,
    J, W, tau,
    Iext, Ibgk, sig,
    sigmaoverride,
    nstats=10,
    stim_area=None,
    stim_intensity=None):
    """
    Simulates interareal dynamics for Rest and Stim conditions.
    Output format is rate[pop, time, area, stat]
    """

    Nt = len(t)
    Nareas = 2
    Npops  = 4

    rate_V_rest = np.zeros((Nareas * Npops, Nt, 1, nstats))
    rate_V_stim = np.zeros((Nareas * Npops, Nt, 1, nstats))

    for s in range(nstats):

        # --------------------
        # 1. REST
        # --------------------
        rate_rest = calculate_rate(
            t, dt, tstop, J, tau, sig,
            Iext, Ibgk, sigmaoverride,
            Nareas, W=W)

        for area in range(Nareas):
            for pop in range(Npops):
                rate_V_rest[area * Npops + pop, :, 0, s] = rate_rest[pop, :, area]

        # --------------------
        # 2. STIM
        # --------------------
        Iext_stim = Iext.copy()
        if stim_area == "V1":
            Iext_stim[0] += stim_intensity # L2/3E in V1
        elif stim_area == "V4":
            Iext_stim[2] += stim_intensity # L5/6E in V4

        rate_stim = calculate_rate(
            t, dt, tstop, J, tau, sig,
            Iext_stim, Ibgk, sigmaoverride,
            Nareas, W=W)

        for area in range(Nareas):
            for pop in range(Npops):
                rate_V_stim[area * Npops + pop, :, 0, s] = rate_stim[pop, :, area]

    return rate_V_rest, rate_V_stim


def trialstat(rate, transient, dt, minfreq_l23, minfreq_l56, nareas, stats):
    '''
    Calculates the periodogram and the peak frequency for L2/3 and L5/6 from the areas
    under analysis.

    Parameters
    ----------
        rate: Simulated rate
        transient: Number of points to average over
        dt: dt of the simulation
        minfrequence_l23: Frequencies below this threshold get discarded. Specific for layer L2/3
        minfrequence_l56: Frequencies below this threshold get discarded. Specific for layer L5/6
        nareas: Number of areas to take into account
        stats: Number or repetions

    '''

    # To obtain some statistics calculate frequency and amplitude for multiple runs

    powerpeak = np.zeros((nareas * nareas, stats))
    freqpeak = np.zeros((nareas * nareas, stats))
    # Calculate periodeogram to get the shape of the array
    pxx_t, fxx_t = calculate_periodogram(rate[0, :, 0, 0], transient, dt)
    px2 = np.zeros((len(pxx_t), nareas, stats))
    fx2 = np.zeros((len(fxx_t), nareas, stats))
    px5 = np.zeros((len(pxx_t), nareas, stats))
    fx5 = np.zeros((len(fxx_t), nareas, stats))

    for stat in range(stats):
        k = 0
        for area in range(nareas):
            # Calculate power spectrum for the excitatory population from layer L23
            # rate shape: (Npops, Nt, Nareas, nstats) where pop 0 = L2E, pop 2 = L5E
            pxx2, fxx2 = calculate_periodogram(rate[area * 4, :, 0, stat], transient, dt)
            # concatenate the results
            px2[:, area, stat] = pxx2
            fx2[:, area, stat] = fxx2

            frequency_l23, amplitudeA_l23, _, _ = \
                find_peak_frequency(fxx2, pxx2, minfreq_l23, rate[area * 4, :, 0, stat])

            powerpeak[k, stat] = amplitudeA_l23
            freqpeak[k, stat] = frequency_l23

            k += 1

            # Calculate power spectrum for the excitatory population from layer L56
            pxx5, fxx5 = calculate_periodogram(rate[area * 4 + 2, :, 0, stat], transient, dt)
            # concatenate the results
            px5[:, area, stat] = pxx5
            fx5[:, area, stat] = fxx5

            frequency_l56, amplitudeA_l56, _, _ = \
                find_peak_frequency(fxx5, pxx5, minfreq_l56, rate[area * 4 + 2, :, 0, stat])

            powerpeak[k, stat] = amplitudeA_l56
            freqpeak[k, stat] = frequency_l56
            k += 1
    return fx2, px2, fx5, px5, powerpeak, freqpeak


def interareal_analysis(rate_rest, rate_stim, transient, dt, minfreq_l23, minfreq_l56, nareas, stats):
    '''
    Calculate the power spectrum and its peak value.

    Parameters
    ----------
        rate_rest: Simulated rest time series
        rate_stim: Simulated stimulated time series
        transient:
        dt: dt of the simulation
        minfrequence_l23: Frequencies below this threshold get discarded. Specific for layer L2/3
        minfrequence_l56: Frequencies below this threshold get discarded. Specific for layer L5/6
        nareas: Number of areas to take into account
        stats: Number or repetions
    Returns
    -------
        px20: Power spectrum for layer L2/3 at rest
        px2: Power spectrum for layer L3/3 after stimulus
        px50: Power spectrum for layer L5/6 at rest
        px5: Power spectrum for layer L5/6 after stimulus
        fx2: Array of sample frequencies
        pgamma: P-value for the difference between stimulus and rest for the gamma frequency
        palpha: P-value for the difference between stimulus and rest for the alpha frequency
    '''
    # Analysis of the simulation at rest
    fx20, px20, fx50, px50, powerpeak0, fpeak0 = trialstat(rate_rest, transient, dt, minfreq_l23, minfreq_l56, nareas, stats)
    # Analysis of the simulation with additional stimulation
    fx2, px2, fx5, px5, powerpeak1, fpeak1 = trialstat(rate_stim, transient, dt, minfreq_l23, minfreq_l56, nareas, stats)

    # Analysis after microstimulation. Test if there is a significant difference in the max peak between
    # rest and stimulation significance
    #  Note: We are using -1 because Python starts with 0 indexing
    z1 = 3-1; z2 = 4-1; # Excitatory and inhibitory layers L5/6 (feedforward)
    gamma0 = powerpeak0[z1, :]; gamma1 = powerpeak1[z1, :]
    alpha0 = powerpeak0[z2, :]; alpha1 = powerpeak1[z2, :]

    statistic, pgamma = ttest_ind(gamma0, gamma1)
    statistic2, palpha = ttest_ind(alpha0, alpha1)

    return px20, px2, px50, px5, fx2, pgamma, palpha


def plot_powerspectrum(recording_area, layer, px0, px, fx2, lcolours, stimulated_area, nstats):
    '''
    Plot power spectrum for the rest and stimulated layers for the areas under analysis

    Parameters
    ----------
        recording_area: Area where the stimulus is being recorded
        layer: Layer of interest (can be either L2/3 or L/56)
        px0: Rate time series with no stimulation
        px: Rate time series with stimulation
        fx2: Array of sample frequencies
        lcolours: List of colours to use for plotting
        nstats: Number of simulation repetitions
    '''

    barrasfrequency = np.zeros((2, 2))

    # Set configuration according to passed layers
    window_size = 99
    if recording_area == 'V1':
        area_idx = 0
    elif recording_area == 'V4':
        area_idx = 1
    else:
        IOError('Rate in this area was not simulated. Please, check your area again.')

    if layer == 'l23':
        resbin = 20
        filter = [20, 80]
        if stimulated_area == 'stimulate_V1':
            ylim = [0, .006]
        elif stimulated_area == 'stimulate_V4':
            ylim = [0, .0015]
    elif layer == 'l56':
        resbin = 10
        filter = [5, 20]
        ylim = [0, .03]
    else:
        IOError('Passed layer is not l23 or l56. Please check your input.')

    ## Analysis for rest model
    # reshuffle the px20 data so that you have stats x time points for the area of interst
    pz0 = np.squeeze(np.transpose(px0[:, area_idx, :]))
    fx0 = np.transpose(fx2[:, area_idx, 0])
    # smooth the data
    # Note: The matlab code transforms an even-window size into an odd number by subtracting by one.
    # So for simplicity I already define the window size as an odd number
    pxx0 = []
    for i in range(nstats):
        pxx0.append(matlab_smooth(pz0[i, :], window_size))
    pxx0 = np.asarray(pxx0)
    pxx20 = np.mean(pxx0, axis=0)
    pxx20sig = np.std(pxx0, axis=0)

    fxx_plt_idx_0 = np.where((fx0 > filter[0]) & (fx0 < filter[1]))
    z10 = pxx20[fxx_plt_idx_0]
    z20 = pxx20sig[fxx_plt_idx_0]
    b20 = np.argmax(z10)
    barrasfrequency[0, 0] = z10[b20]
    barrasfrequency[0, 1] = z20[b20]

    ## Analysis for model with stimulus
    pz = np.squeeze(np.transpose(px[:, area_idx, :]))
    fx = np.transpose(fx2[:, area_idx, 0])
    pxx = []
    for i in range(nstats):
        pxx.append(matlab_smooth(pz[i, :], window_size))
    pxx = np.asarray(pxx)
    pxx2 = np.mean(pxx, axis=0)
    pxx2sig = np.std(pxx, axis=0)
    fxx_plt_idx = np.where((fx > filter[0]) & (fx < filter[1]))
    z1 = pxx2[fxx_plt_idx]
    z2 = pxx2sig[fxx_plt_idx]
    b2 = np.argmax(z1)
    barrasfrequency[1, 0] = z1[b2]
    barrasfrequency[1, 1] = z2[b2]

    # Plot results
    fig, ax = plt.subplots(1)
    plt_filled_std(ax, fx0[1:-1:resbin], pxx20[1:-1:resbin], pxx20sig[1:-1:resbin], lcolours[0], 'rest')
    plt_filled_std(ax, fx[1:-1:resbin], pxx2[1:-1:resbin], pxx2sig[1:-1:resbin], lcolours[1], 'stimulus')
    plt.xlim(filter)
    plt.ylim(ylim)
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('%s %s Power' %(recording_area, layer))
    plt.legend()
    plt.savefig('interareal/%s_layer_%s_%s.png' %(stimulated_area, recording_area, layer))

    return barrasfrequency


def interareal_plt(areas, px20, px2, px50, px5, fx2, stimulated_area, nstats):
    '''
    Plot Powerspectrum and peak value of the power for the the passed areas of interest

    Parameters
    ----------
        areas: List of areas being simulated
        px20: Rate time series for E L2/3 at rest
        px2: Rate time series for E L2/3 after stimulation
        px50: Rate time series for I L5/6 at rest
        px5: Rate time series for I L5/6 after stimulation
        fx2: Array of sample frequencies
        stimulated_area: Define in which area the stimulus was applied
        nstats: Number of times that the experiment was repeated
    '''

    if stimulated_area == 'stimulate_V4':
        recording_area = areas[0]
    elif stimulated_area == 'stimulate_V1':
        recording_area = areas[1]
    else:
        IOError('Stimulated area is not defined.')

    # Define colours to be used for the plotting
    lcolours = ['#1F5E43', '#31C522', '#944610', '#E67E22']

    # Plot power spectrum for the L23 and L5/6 Layers
    barrasgamma = plot_powerspectrum(recording_area, 'l23', px20, px2, fx2, lcolours[:2], stimulated_area, nstats)
    barrasalpha = plot_powerspectrum(recording_area, 'l56', px50, px5, fx2, lcolours[-2:], stimulated_area, nstats)

    # Plot the peak value of the power spectraum at the supergranular layer for both areas
    if stimulated_area == 'stimulate_V4':
        yaxis_gamma = [0, .002]
        yaxis_alpha = [0, .04]
    elif stimulated_area == 'stimulate_V1':
        yaxis_gamma = [0, .007]
        yaxis_alpha = [0, .04]

    plt.figure()
    plt.bar(['Rest', 'Stim'], barrasgamma[:, 0], color=lcolours[:2], yerr=barrasgamma[:, 1])
    plt.ylim(yaxis_gamma)
    plt.ylabel(r'$\gamma$ power')
    plt.savefig('interareal/%s_layer_gamma.png' % (stimulated_area))

    plt.figure()
    plt.bar(['Rest', 'Stim'], barrasalpha[:, 0], color=lcolours[-2:], yerr=barrasalpha[:, 1])
    plt.ylabel(r'$\alpha$ power')
    plt.ylim(yaxis_alpha)
    plt.savefig('interareal/%s_layer_alpha.png' % (stimulated_area))
