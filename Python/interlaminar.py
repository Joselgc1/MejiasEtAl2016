import numpy as np
import os
import math
import pickle
from scipy import signal, fftpack
import matplotlib.pylab as plt
from neurodsp import spectral
from scipy.io import loadmat

from calculate_rate import calculate_rate
from helper_functions import calculate_periodogram, compress_data, find_peak_frequency


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def my_pretransformations(x, window, noverlap, fs):

    # Place x into columns and return the corresponding central time estimates
    # restructure the data
    ncol = int(np.floor((x.shape[0] - noverlap) / (window.shape[0] - noverlap)))
    coloffsets = np.expand_dims(range(ncol), axis=0) * (window.shape[0] - noverlap)
    rowindices = np.expand_dims(range(0, window.shape[0]), axis=1)

    # segment x into individual columns with the proper offsets
    xin = x[rowindices + coloffsets]
    # return time vectors
    t = coloffsets + (window.shape[0]/2)/ fs
    return xin, t

def compute_goertzel(target_frequency, sampling_rate, data):

    # Number of sample points
    nsamples = data.shape[0]
    scaling_factor = nsamples / 2.0
    k = (.5 + ((nsamples * target_frequency) / sampling_rate))
    omega = (2 * math.pi * k) / nsamples
    sine = math.sin(omega)
    cosine = math.cos(omega)
    coeff = 2 * cosine
    q1 = 0; q2 = 0

    for i in range(nsamples):
        q0 = coeff * q1 - q2 + data[i]
        q2 = q1
        q1 = q0

    real = (q1 - q2 * cosine) / scaling_factor
    imag = (q2 * sine) / scaling_factor
    magnitude = np.sqrt(real * real + imag * imag)
    return magnitude

def goertzel_second(x, k, N):
    # k = k1/ 5000 * 821
    w = 2 * math.pi * k/ N
    cw = math.cos(w); c = 2 * cw
    sw = math.sin(w)
    z1 = 0; z2= 0
    for n in range(N):
        z0 = x[n] + c * z1 - z2
        z2 = z1
        z1 = z0
    real = cw * z1 - z2
    imag = sw * z1
    return complex(real, imag)

def goertzel_third(x, k1, N, f, Fs):
    k = k1/ 5000 * 821
    w = (2 * math.pi * f)/ (Fs * N**2)
    cw = math.cos(w); c = 2 * cw
    sw = math.sin(w)
    z1 = 0
    z2= 0
    for n in range(N):
        z0 = x[n] + c * z1 - z2
        z2 = z1
        z1 = z0
    real = cw * z1 - z2
    imag = sw * z1
    return complex(real, imag)


def interlaminar_simulation(
    analysis, 
    t, 
    dt, 
    tstop, 
    J, 
    tau, 
    sig, 
    Iext, 
    Ibgk, 
    noise, 
    Nareas):

    rate = calculate_rate(t, dt, tstop, J, tau, sig, Iext, Ibgk, noise, Nareas)

    picklename = os.path.join(analysis, 'simulation.pckl')
    with open(picklename, 'wb') as filename:
        pickle.dump(rate, filename)

    print('    Done Simulation!')
    return rate

def interlaminar_activity_analysis(rate, transient, dt, t, min_freq5):

    # Note: This analysis selects only the excitatory populations from L2/3 and L5/6

    # Extract firing rates
    x_2 = rate[0, int(round((transient + dt)/dt)) - 1:, 0]
    x_5 = rate[2, int(round((transient + dt)/dt)) - 1:, 0]

    # Compute power spectrum of L5
    pxx, fxx = calculate_periodogram(x_5, transient, dt)
    f_peakalpha, _, _, _ = find_peak_frequency(fxx, pxx, min_freq5, x_5)
    print('    Average peak frequency on the alpha range: %.02f Hz' %f_peakalpha)

    # Bandpass filter around alpha
    fs = 1/dt
    fmin = 7 
    fmax = 12
    filter_order = 3
    bf, af = signal.butter(filter_order, [fmin/(fs/2), fmax/(fs/2)], 'bandpass') # Note: padlen is differently defined in the scipy implementation
    re5bp = -signal.filtfilt(bf, af, x_5, padlen=3*(max(len(af), len(bf)) - 1)) # simulated LFP

    # Divide into zones of fixed length
    tzone = 4
    tzoneindex = int(round((tzone/dt))) # length of the tzone in indices

    rest = len(t) % tzoneindex
    time5 = t[0:-rest]
    re5 = re5bp[0:-rest]
    re2 = x_2[0:-rest]

    numberofzones = int(round(len(re5)/tzoneindex))
    zones5 = np.reshape(re5, (tzoneindex, numberofzones), order='F')
    zones2 = np.reshape(re2, (tzoneindex, numberofzones), order='F') # for re2

    # Find alpha peaks inside each zone
    tzi_bottom = int(round(tzoneindex/2 - tzoneindex/4)) + 1
    tzi_top = int(round(tzoneindex/2 + tzoneindex/4)) + 1

    alpha_peaks = np.zeros((numberofzones))
    aploc = np.zeros((numberofzones))

    # Find max value for each zone
    for i in range(numberofzones):
        segment = zones5[tzi_bottom:tzi_top, i]
        alpha_peaks[i] = np.max(segment)
        aploc[i] = np.argmax(segment) + tzi_bottom

    # Segment around alpha peaks
    seglength = 7/f_peakalpha
    if seglength/2 >= tzi_bottom * dt:
        print('Problems with segment window!')
    segindex = int(round(0.5 * seglength / dt)) # segment semi-length in indices
    
    segind01 = int(round(aploc[0] - segindex) + 2)
    segind02 = int(round(aploc[0] + segindex) + 2)
    segment2 = np.zeros((segind02 - segind01 + 1, numberofzones))
    segment5 = np.zeros((segind02 - segind01 + 1, numberofzones))
    
    for i in range(numberofzones):
        segind1 = int(round(aploc[i] - segindex) + 2)
        segind2 = int(round(aploc[i] + segindex) + 2)
        if alpha_peaks[i] >= 0.:
            segment5[:, i] = zones5[segind1:segind2 + 1, i]
            segment2[:, i] = zones2[segind1:segind2 + 1, i]

    return segment5, segment2, segindex, numberofzones

def interlaminar_analysis_periodeogram(rate, segment2, transient, dt, min_freq2, numberofzones):

    # Compute gamma peak frequency
    restate = rate[0, :, 0]
    pxx2, fxx2 = calculate_periodogram(restate, transient, dt)
    f_peakgamma, _, _, _ = find_peak_frequency(fxx2, pxx2, min_freq2, restate)
    print('    Average peak frequency on the gamma range: %.02f Hz' %f_peakgamma)

    timewindow = 7/f_peakgamma
    window_len = int(round(timewindow/dt))
    window = signal.get_window('hamming', window_len)
    noverlap = int(round(0.95 * window_len))
    lowest_frequency = 25
    highest_frequency = 45
    step_frequency = .25
    freq_displayed = np.arange(lowest_frequency, highest_frequency + step_frequency, step_frequency)
    fs = int(1/dt) # sampling frequency

    dic = {
        'segment2': segment2,
        'window_len': window_len,
        'noverlap': noverlap,
        'freq_displayed': freq_displayed,
        'fs': fs
    }
    with open('periodogram.pckl', 'wb') as filename:
        pickle.dump(dic, filename)

    Sxx = np.zeros((freq_displayed.shape[0], 83, numberofzones), dtype=complex)
    for n in range(numberofzones):
        xin, t = my_pretransformations(segment2[:, n], window, noverlap, fs)
        data = np.multiply(np.expand_dims(window, axis=1), xin)

        for jj in range(data.shape[1]):
            for ii in range(freq_displayed.shape[0]):
                Sxx[ii, jj, n] = goertzel_second(data[:, jj], freq_displayed[ii], data.shape[0])

    Sxx_mean = np.mean(Sxx, axis=2)
    U = np.dot(np.expand_dims(window, axis=0), window)
    Sxx_conj = (Sxx_mean * np.conj(Sxx_mean))/U
    Sxx_fin = Sxx_conj.astype(float)

    goertzel_second(data[:, 0], freq_displayed[0], data.shape[0])
    goerzel = compute_goertzel(freq_displayed[0], fs, data[:, 0])
    # TODO: Try to use fft instead of the goertzel algorithm to calculate the fft
    Xx = np.zeros((freq_displayed.shape[0], xin.shape[1], numberofzones), dtype=complex)
    for n in range(numberofzones):
        for i in range(xin.shape[1]):
            Xx[:, i, n] = fftpack.fft(data[:, i], freq_displayed.shape[0])

        ff, tt, Sxx = signal.spectrogram(
            segment2[:, i], fs=fs, window=window, noverlap=noverlap, 
            return_onesided=False, detrend=False, scaling='density', mode='psd'
        )

    print('Done Analysis!')
    return ff, tt, Sxx

def calculate_interlaminar_power_spectrum(rate, dt, transient, Nbin):
    # Calculate the rate for the passed connectivity
    pxx_l23, fxx_l23 = calculate_periodogram(rate[0, :, 0], transient, dt)
    pxx_l56, fxx_l56 = calculate_periodogram(rate[2, :, 0], transient, dt)
    # Compress data by selecting one data point every "bin"
    pxx_l23_bin, fxx_l23_bin = compress_data(pxx_l23, fxx_l23, Nbin)
    pxx_l56_bin, fxx_l56_bin = compress_data(pxx_l56, fxx_l56, Nbin)
    return pxx_l23_bin, fxx_l23_bin, pxx_l56_bin, fxx_l56_bin

def interlaminar_3C_analysis(
    t, dt, tstop,
    J, tau, sig,
    base_Iext, Ibgk,
    sigmaoverride,
    Nareas,
    sweep_I_L56E,
    nruns,
    transient):

    results = {
        "I_L56E": [],
        "L23_gamma_power_mean": [],
        "L23_gamma_power_std": [],
        "L23_rate_mean": [],
        "L23_rate_std": [],
        "L56_alpha_power_mean": [],
        "L56_alpha_power_std": []
    }

    fs = 1/dt
    trans_idx = int(transient / dt)

    for I_L56E in sweep_I_L56E:
        print(" Sweeping:", I_L56E)

        gamma_powers = []
        gamma_rates  = []
        alpha_powers = []

        for run in range(nruns):

            # Build external input vector
            Iext_vec = np.array(base_Iext, dtype=float)
            Iext_vec[2] = I_L56E    # L5/6E only

            rate = calculate_rate(
                t, dt, tstop,
                J, tau, sig,
                Iext_vec, Ibgk,
                sigmaoverride,
                Nareas
            )

            r23 = rate[0, trans_idx:, 0]
            r56 = rate[2, trans_idx:, 0]

            # L23 mean firing rate (panel 2)
            gamma_rates.append(np.mean(r23))

            # L23 gamma power (panel 1)
            pxx23, fxx23 = calculate_periodogram(r23, 0, dt)
            # fxx23, pxx23 = signal.periodogram(r23, fs=fs)
            gamma_band = pxx23[(fxx23 >= 30) & (fxx23 <= 80)]
            gamma_powers.append(np.max(gamma_band))

            # L56 alpha power (panel 3)
            pxx56, fxx56 = calculate_periodogram(r56, 0, dt)
            # fxx56, pxx56 = signal.periodogram(r56, fs=fs)
            alpha_band = pxx56[(fxx56 >= 5) & (fxx56 <= 30)]
            alpha_powers.append(np.max(alpha_band))

        # store aggregated results
        results["I_L56E"].append(I_L56E)
        results["L23_gamma_power_mean"].append(np.mean(gamma_powers))
        results["L23_gamma_power_std"].append(np.std(gamma_powers))
        results["L23_rate_mean"].append(np.mean(gamma_rates))
        results["L23_rate_std"].append(np.std(gamma_rates))
        results["L56_alpha_power_mean"].append(np.mean(alpha_powers))
        results["L56_alpha_power_std"].append(np.std(alpha_powers))

    # convert lists to arrays
    for k in results.keys():
        results[k] = np.array(results[k])

    return results


def plot_spectrogram(ff, tt, Sxx):
    plt.figure()
    plt.pcolormesh(tt, ff, Sxx, cmap='jet')
    plt.ylim([25, 45])
    plt.show()

def plot_power_spectrum_neurodsp(dt, rate_conn, rate_noconn, analysis):
    fs = 1/dt

    # Plot the results for L23
    freq_mean_L23_conn, P_mean_L23_conn = spectral.compute_spectrum(rate_conn[0, :, 0], fs, avg_type='mean')
    freq_mean_L23_noconn, P_mean_L23_noconn = spectral.compute_spectrum(rate_noconn[0, :, 0], fs, avg_type='mean')

    plt.figure()
    plt.loglog(freq_mean_L23_conn, P_mean_L23_conn, label='Coupled', linewidth=2, color='g')
    plt.loglog(freq_mean_L23_noconn, P_mean_L23_noconn, label='Uncoupled', linewidth=2, color='k')
    plt.xlim([1, 100])
    plt.ylim([10 ** -4, 10 ** -2])
    plt.ylabel('Power')
    plt.xlabel('Frequency (Hz)')
    plt.legend()

    # Plot the results for L56
    freq_mean_L56_conn, P_mean_L56_conn = spectral.compute_spectrum(rate_conn[2, :, 0], fs, avg_type='mean')
    freq_mean_L56_noconn, P_mean_L56_noconn = spectral.compute_spectrum(rate_noconn[2, :, 0], fs, avg_type='mean')

    plt.figure()
    plt.loglog(freq_mean_L56_conn, P_mean_L56_conn, label='Coupled', linewidth=2, color='#FF7F50')
    plt.loglog(freq_mean_L56_noconn, P_mean_L56_noconn, label='Uncoupled', linewidth=2, color='k')
    plt.xlim([1, 100])
    plt.ylim([10**-5, 10**-0])
    plt.ylabel('Power')
    plt.xlabel('Frequency (Hz)')
    plt.legend()

def plot_interlaminar_power_spectrum(
    fxx_uncoupled_l23_bin, fxx_coupled_l23_bin,
    pxx_uncoupled_l23_bin, pxx_coupled_l23_bin,
    fxx_uncoupled_l56_bin, fxx_coupled_l56_bin,
    pxx_uncoupled_l56_bin, pxx_coupled_l56_bin,
    analysis):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7))

    # Plot the results for L23 (top panel)
    ax1.loglog(fxx_uncoupled_l23_bin, pxx_uncoupled_l23_bin, 'k', label='Uncoupled')
    ax1.loglog(fxx_coupled_l23_bin, pxx_coupled_l23_bin, 'g', label='Coupled')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('L2/3 power')
    ax1.legend(frameon=False)
    ax1.set_xlim([1, 100])
    ax1.set_ylim([10**-4, 10**-2])

    # Plot the results for L56 (bottom panel)
    ax2.loglog(fxx_uncoupled_l56_bin, pxx_uncoupled_l56_bin, 'k', label='Uncoupled')
    ax2.loglog(fxx_coupled_l56_bin, pxx_coupled_l56_bin, color='#FF7F00', label='Coupled')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('L5/6 power')
    ax2.legend(frameon=False)
    ax2.set_xlim([1, 100])
    ax2.set_ylim([10**-5, 10**0])

    plt.tight_layout()
    if not os.path.exists(analysis):
        os.makedirs(analysis)
    plt.savefig(os.path.join(analysis, 'power_spectrum.png'), dpi=150, bbox_inches='tight')

def plot_activity_traces(dt, segment5, segindex, analysis):
    # calculate the peak-centered alpha wave by averaging
    alphawaves = np.mean(segment5, axis=1)
    alphatime = [(i*dt) - (segindex*dt) for i in range(1, alphawaves.shape[0] + 1)]
    # plot the first 100 elements from segment5
    grey_rgb = (.7, .7, .7)
    plt.figure()
    plt.plot(alphatime, segment5[:, 0:100], color=grey_rgb)
    plt.plot(alphatime, alphawaves, 'b')
    plt.xlabel('Time relative to alpha peak (s)')
    plt.ylabel('LFP, L5/6')
    plt.xlim([-.24, .24])
    plt.savefig(os.path.join(analysis, 'activity_traces.png'))

def interlaminar_3C_plot(results, analysis_name="interlaminar_c"):

    if not os.path.exists(analysis_name):
        os.makedirs(analysis_name)

    I = results["I_L56E"]

    gamma     = results["L23_gamma_power_mean"]
    gamma_std = results["L23_gamma_power_std"]
    rate      = results["L23_rate_mean"]
    rate_std  = results["L23_rate_std"]
    alpha     = results["L56_alpha_power_mean"]
    alpha_std = results["L56_alpha_power_std"]

    fig, axs = plt.subplots(2, 2, figsize=(9, 7))

    # Panel 1 — Gamma power vs L5E input
    ax = axs[0, 0]
    ax.plot(I, gamma, color="green")
    ax.fill_between(I, gamma-gamma_std, gamma+gamma_std, alpha=0.3, color="green")
    ax.set_ylabel("L2/3E γ power")
    ax.set_xlabel("Input to L5/6E")
    ax.set_xlim([min(I), max(I)])

    # Panel 2 — Firing rate vs L5E input
    ax = axs[0, 1]
    ax.plot(I, rate, color="blue")
    ax.fill_between(I, rate-rate_std, rate+rate_std, alpha=0.3, color="blue")
    ax.set_ylabel("L2/3E firing rate")
    ax.set_xlabel("Input to L5/6E")
    ax.set_xlim([min(I), max(I)])

    # Panel 3 — Alpha power vs L5E input
    ax = axs[1, 0]
    ax.plot(I, alpha, color="orange")
    ax.fill_between(I, alpha-alpha_std, alpha+alpha_std, alpha=0.3, color="orange")
    ax.set_xlabel("Input to L5/6E")
    ax.set_ylabel("L5/6E α power")
    ax.set_xlim([min(I), max(I)])

    # Panel 4 — L5 alpha vs L2/3 firing rate
    ax = axs[1, 1]
    order = np.argsort(rate)   # Makes curve smooth
    ax.plot(rate[order], alpha[order], color="gold")
    ax.fill_between(rate[order],
                    alpha[order] - alpha_std[order],
                    alpha[order] + alpha_std[order],
                    color="gold", alpha=0.3)
    ax.set_xlabel("L2/3E firing rate")
    ax.set_ylabel("L5/6E α power")

    # OPTIONAL: you can specify xlim manually if needed
    # ax.set_xlim([min(rate)*0.9, max(rate)*1.1])

    plt.tight_layout()
    plt.savefig(os.path.join(analysis_name, "interlaminar_3C.png"))
    plt.close(fig)

