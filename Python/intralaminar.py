import os
import numpy as np
import pickle
from scipy import signal
import matplotlib.pylab as plt
from matplotlib.ticker import ScalarFormatter

from calculate_rate import calculate_rate
from helper_functions import calculate_periodogram, compress_data, plt_filled_std, matlab_smooth

def intralaminar_simulation(
    analysis, 
    layer, 
    Iexts, 
    Ibgk, 
    nruns, 
    t, 
    dt, 
    tstop,
    J, 
    tau, 
    sig, 
    noise, 
    Nareas):

    simulation = {}
    for Iext in Iexts:
        simulation[Iext] = {}
        # Input vector for the excitatory and inhibitory populations
        # Meaning:
        # L23 excitatory gets stimulus
        # L23 inhibitory does NOT
        # L5 excitatory gets stimulus
        # L5 inhibitory does NOT
        Iext_a = np.array([Iext, 0, Iext, 0])

        # run each combination of external input multiple times an take the average PSD
        for nrun in range(nruns):

            simulation[Iext][nrun] = {}
            rate = calculate_rate(t, dt, tstop, J, tau, sig, Iext_a, Ibgk, noise, Nareas)

            # Note: Save only the excitatory and inhibitory signal from L2/3.
            # For compatibility with NeuroML/LEMS transform the results into a row matrix
            simulation[Iext][nrun]['L23_E/0/L23_E/r'] = rate[0, :].reshape(-1)
            simulation[Iext][nrun]['L23_I/0/L23_I/r'] = rate[1, :].reshape(-1)
            simulation[Iext][nrun]['L5_E/0/L5_E/r'] = rate[2, :].reshape(-1)
            simulation[Iext][nrun]['L5_I/0/L5_I/r'] = rate[3, :].reshape(-1)


    if not os.path.exists(analysis):
        os.makedirs(analysis, exist_ok=True)
    picklename = os.path.join(analysis, layer + '_simulation.pckl')
    with open(picklename, 'wb') as file1:
        pickle.dump(simulation, file1)
    print('    Done Simulation!')
    return simulation

def intralaminar_analysis(
    simulation, 
    Iexts, 
    nruns, 
    layer='L23', 
    dt=2e-04, 
    transient=5):
    """
    Calculates the main intralaminar analysis and dumps a pickle containing the periodogram of the analysis
    Inputs
        simulation: dictionary containing all the simulations to be analysed
        Iexts: a list of the input strengths applied on the excitatory populations
        nruns: number of simulations analysed for every Iext
        layer: Layer under analysis
        dt: time step of the simulation
        transient: transient time of the simulation
        Returns:
            psd_dic: dictionary containing the periodogram of the analysis
    """

    psd_dic = {}

    for Iext in Iexts:
        psd_dic[Iext] = {}

        for nrun in range(nruns):

            psd_dic[Iext][nrun] = {}
            restate = simulation[Iext][nrun]['L23_E/0/L23_E/r']

            # perform periodogram on restate.
            pxx2, fxx2 = calculate_periodogram(restate, transient, dt)

            # Compress the data by sampling every 5 points.
            bin_size = 5
            pxx_bin, fxx_bin = compress_data(pxx2, fxx2, bin_size)

            # smooth the data
            # Note: The matlab code transforms an even-window size into an odd number by subtracting by one.
            # So for simplicity I already define the window size as an odd number
            window_size = 79
            pxx = matlab_smooth(pxx_bin, window_size)

            psd_dic[Iext][nrun]['pxx'] = pxx

        # take the mean and std over the different runs
        psd_dic[Iext]['mean_pxx'] = np.mean([psd_dic[Iext][i]['pxx'] for i in range(nruns)], axis=0)
        psd_dic[Iext]['std_pxx'] = np.std([psd_dic[Iext][i]['pxx'] for i in range(nruns)], axis=0)

    # add fxx_bin to dictionary
    psd_dic['fxx_bin'] = fxx_bin

    print('    Done Analysis!')
    return psd_dic

def intralaminar_peak_analysis(
    simulation, 
    Iexts, 
    nruns, 
    layer='L23', 
    dt=2e-4, 
    transient=5):

    # store results (mean + std)
    results = {
        'L23': {
            'Iext': [], 
            'peak_power_mean': [], 
            'peak_power_std': [],
            'peak_freq_mean': [],  
            'peak_freq_std': []
        },
        'L5': {
            'Iext': [], 
            'peak_power_mean': [], 
            'peak_power_std': [],
            'peak_freq_mean': [],  
            'peak_freq_std': []
        }
    }

    # baseline_L23 = None
    # baseline_L5  = None
    # if 0 in Iexts:
    #     all_pxx_L23_0 = []
    #     all_pxx_L5_0  = []
    #     for nrun in range(nruns):
    #         r23 = simulation[0][nrun]['L23_E/0/L23_E/r']
    #         r5  = simulation[0][nrun]['L5_E/0/L5_E/r']
    #         pxx23, fxx = calculate_periodogram(r23, transient, dt)
    #         pxx5,  _   = calculate_periodogram(r5, transient, dt)
    #         all_pxx_L23_0.append(pxx23)
    #         all_pxx_L5_0.append(pxx5)
    #     baseline_L23 = np.mean(all_pxx_L23_0, axis=0)
    #     baseline_L5  = np.mean(all_pxx_L5_0,  axis=0)

    for Iext in Iexts:

        L23_peak_powers = []
        L23_peak_freqs  = []
        L5_peak_powers  = []
        L5_peak_freqs   = []

        for nrun in range(nruns):
            r23 = simulation[Iext][nrun]['L23_E/0/L23_E/r']
            r5  = simulation[Iext][nrun]['L5_E/0/L5_E/r']
            
            # perform periodogram on restate.
            pxx23, fxx = calculate_periodogram(r23, transient, dt)
            pxx5,  _   = calculate_periodogram(r5, transient, dt)

            # # subtract the baseline from the periodogram
            # if baseline_L23 is not None:
            #     pxx23 = pxx23 - baseline_L23
            #     pxx5  = pxx5  - baseline_L5

            # find the mask for the gamma and alpha bands
            gamma_mask = (fxx >= 30) & (fxx <= 80)
            alpha_mask = (fxx >= 5) & (fxx <= 30)

            # get the power and frequency of the gamma and alpha peaks
            gamma_pxx = pxx23[gamma_mask]
            gamma_f   = fxx[gamma_mask]
            alpha_pxx = pxx5[alpha_mask]
            alpha_f   = fxx[alpha_mask]

            # find the index of the gamma and alpha peaks
            idx_g = np.argmax(gamma_pxx)
            idx_a = np.argmax(alpha_pxx)

            # append the power and frequency of the gamma and alpha peaks
            L23_peak_powers.append(gamma_pxx[idx_g])
            L23_peak_freqs.append(gamma_f[idx_g])
            L5_peak_powers.append(alpha_pxx[idx_a])
            L5_peak_freqs.append(alpha_f[idx_a])

        # append means and stds
        results['L23']['Iext'].append(Iext)
        results['L23']['peak_power_mean'].append(np.mean(L23_peak_powers))
        results['L23']['peak_power_std'].append(np.std(L23_peak_powers))
        results['L23']['peak_freq_mean'].append(np.mean(L23_peak_freqs))
        results['L23']['peak_freq_std'].append(np.std(L23_peak_freqs))

        results['L5']['Iext'].append(Iext)
        results['L5']['peak_power_mean'].append(np.mean(L5_peak_powers))
        results['L5']['peak_power_std'].append(np.std(L5_peak_powers))
        results['L5']['peak_freq_mean'].append(np.mean(L5_peak_freqs))
        results['L5']['peak_freq_std'].append(np.std(L5_peak_freqs))

    return results

def intralaminar_plt(psd_dic, output_dir='intralaminar'):
    # select only the first time points until fxx < 100
    fxx_plt_idx = np.where(psd_dic['fxx_bin'] < 100)
    fxx_plt = psd_dic['fxx_bin'][fxx_plt_idx]

    # find the correspondent mean and std pxx for this range
    Iexts = list(psd_dic.keys())
    # remove the fxx_bin key
    if 'fxx_bin' in Iexts:
        Iexts.remove('fxx_bin')
    for Iext in Iexts:
        psd_dic[Iext]['mean_pxx'] = psd_dic[Iext]['mean_pxx'][fxx_plt_idx]
        psd_dic[Iext]['std_pxx'] = psd_dic[Iext]['std_pxx'][fxx_plt_idx]

    # find the difference regarding the no_input
    psd_mean_0_2 = psd_dic[2]['mean_pxx'] - psd_dic[0]['mean_pxx']
    psd_mean_0_4 = psd_dic[4]['mean_pxx'] - psd_dic[0]['mean_pxx']
    psd_mean_0_6 = psd_dic[6]['mean_pxx'] - psd_dic[0]['mean_pxx']

    # find the std
    psd_std_0_2 = np.sqrt(psd_dic[2]['std_pxx'] ** 2 + psd_dic[0]['std_pxx'] ** 2)
    psd_std_0_4 = np.sqrt(psd_dic[4]['std_pxx'] ** 2 + psd_dic[0]['std_pxx'] ** 2)
    psd_std_0_6 = np.sqrt(psd_dic[6]['std_pxx'] ** 2 + psd_dic[0]['std_pxx'] ** 2)

    lcolours = ['#588ef3', '#f35858', '#bd58f3']
    fig, ax = plt.subplots(1)
    plt_filled_std(ax, fxx_plt, psd_mean_0_2, psd_std_0_2, lcolours[0], 'Input = 2')
    plt_filled_std(ax, fxx_plt, psd_mean_0_4, psd_std_0_4, lcolours[1], 'Input = 4')
    plt_filled_std(ax, fxx_plt, psd_mean_0_6, psd_std_0_6, lcolours[2], 'Input = 6')
    plt.xlim([10, 80])
    plt.ylim([0, 0.003])
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Power (resp. rest)')
    plt.legend()

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, -3))
    ax.yaxis.set_major_formatter(formatter)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, 'intralaminar_2B.png'))

def intralaminar_peak_plt(results, output_dir='intralaminar'):
    Iexts = np.array(results['L23']['Iext'])

    alpha = 0.3
    fig, axs = plt.subplots(2, 2, figsize=(8,6), sharex='col')

    # ----- L23 POWER -----
    mean = np.array(results['L23']['peak_power_mean'])
    std  = np.array(results['L23']['peak_power_std'])
    axs[0,0].plot(Iexts, mean, color='tab:green')
    axs[0,0].fill_between(Iexts, mean-std, mean+std, alpha=alpha, color='tab:green')
    axs[0,0].set_ylabel('Gamma peak power (L2/3)')

    # ----- L23 FREQ -----
    mean = np.array(results['L23']['peak_freq_mean'])
    std  = np.array(results['L23']['peak_freq_std'])
    axs[0,1].plot(Iexts, mean, color='tab:cyan')
    axs[0,1].fill_between(Iexts, mean-std, mean+std, alpha=alpha, color='tab:cyan')
    axs[0,1].set_ylabel('Gamma peak freq (Hz)')

    # ----- L5 POWER -----
    mean = np.array(results['L5']['peak_power_mean'])
    std  = np.array(results['L5']['peak_power_std'])
    axs[1,0].plot(Iexts, mean, color='tab:orange')
    axs[1,0].fill_between(Iexts, mean-std, mean+std, alpha=alpha, color='tab:orange')
    axs[1,0].set_xlabel('Input to E population')
    axs[1,0].set_ylabel('Alpha peak power (L5)')

    # ----- L5 FREQ -----
    mean = np.array(results['L5']['peak_freq_mean'])
    std  = np.array(results['L5']['peak_freq_std'])
    axs[1,1].plot(Iexts, mean, color='gold')
    axs[1,1].fill_between(Iexts, mean-std, mean+std, alpha=alpha, color='gold')
    axs[1,1].set_xlabel('Input to E population')
    axs[1,1].set_ylabel('Alpha peak freq (Hz)')

    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'intralaminar_2C.png'))
