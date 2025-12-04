from __future__ import print_function, division

import os
import numpy as np
import argparse
import matplotlib.pylab as plt
import pickle

# set random set
np.random.RandomState(seed=42)

from intralaminar import intralaminar_simulation, intralaminar_analysis, intralaminar_plt, intralaminar_peak_analysis, intralaminar_peak_plt
from interlaminar import interlaminar_simulation, interlaminar_activity_analysis, plot_activity_traces, \
                         calculate_interlaminar_power_spectrum, interlaminar_analysis_periodeogram, \
                         plot_interlaminar_power_spectrum, plot_power_spectrum_neurodsp, \
                        interlaminar_3C_analysis, interlaminar_3C_plot
from interareal import build_interareal_W, interareal_simulation, interareal_analysis, interareal_plt

from helper_functions import firing_rate_analysis, get_network_configuration


"""
Main Python file that contains the definitions for the simulation and
calls the necessary functions depending on the passed parameters.
"""

def getArguments():
    parser = argparse.ArgumentParser(description='Parameters for the simulation')
    parser.add_argument('-sigmaoverride', type=float, dest='sigmaoverride', default=None, help='Override sigma of the Gaussian noise for ALL populations (if None leave them as is)')
    parser.add_argument('-analysis', type=str, dest='analysis', default='debug', help='Specify type of analysis to be used')
    parser.add_argument('-debug', dest='debug', action='store_true', help='Specify whether to generate simulations for debugging')
    parser.add_argument('-noconns', dest='noconns', action='store_true', help='Specify whether to remove connections (DEBUG MODE ONLY!)')
    parser.add_argument('-testduration', type=float, dest='testduration', default=1000., help='Duration of test simulation (DEBUG MODE ONLY!)')
    parser.add_argument('-dt', type=float, dest='dt', default=2e-4, help='Timestep (dt) of simulation in seconds')
    parser.add_argument('-initialrate', type=float, dest='initialrate', default=-1, help='Initial rate of test simulation, if negative, use a random value (default) (DEBUG MODE ONLY!)')
    parser.add_argument('-nogui', dest='nogui', action='store_true', help='No gui')
    return parser.parse_args()

if __name__ == "__main__":
    args = getArguments()

    # Create folder where results will be saved
    if not os.path.isdir(args.analysis):
        os.mkdir(args.analysis)

    if args.analysis == 'intralaminar':
        print('-----------------------')
        print('Intralaminar Analysis')
        print('-----------------------')

        # Time parameters
        dt = args.dt
        tstop = 25        # seconds
        t = np.linspace(0, int(tstop), int(tstop/dt))
        transient = 5

        # Number of areas (intralaminar = 1)
        Nareas = 1

        # Load network config
        tau, sig, J, Iexts, Ibgk = get_network_configuration('intralaminar', noconns=False)
        nruns = 10

        layer = 'L23'
        print(f'    Analysing layer {layer}')

        # Check if simulation already exists
        simulation_file = 'intralaminar/L23_simulation.pckl'
        if not os.path.isfile(simulation_file):
            print('    Re-calculating the simulation...')
            simulation = intralaminar_simulation(
                args.analysis, layer, Iexts, Ibgk, nruns,
                t, dt, tstop, J, tau, sig, args.sigmaoverride, Nareas
            )
        else:
            print(f'    Loading the pre-saved simulation file: {simulation_file}')
            with open(simulation_file, 'rb') as f:
                simulation = pickle.load(f)

        # -------------------------------
        #   FIGURE 2B – Contrast → PSD
        # -------------------------------
        print('    Running PSD analysis for Fig 2B...')
        psd_analysis = intralaminar_analysis(
            simulation, Iexts, nruns, layer, dt, transient
        )
        intralaminar_plt(psd_analysis)
        print('    Saved: intralaminar/intralaminar_2B.png')

        # -------------------------------
        #   FIGURE 2C – Peak Power / Frequency
        # -------------------------------
        print('    Running peak-power & peak-frequency analysis for Fig 2C...')
        peak_results = intralaminar_peak_analysis(
            simulation, Iexts, nruns, layer, dt, transient
        )
        intralaminar_peak_plt(peak_results)
        print('    Saved: intralaminar/intralaminar_2C.png')

        print('Intralaminar analysis completed.')

    if args.analysis == 'interlaminar_a':
        print('-----------------------')
        print('Interlaminar Analysis')
        print('-----------------------')

        # Calculates the power spectrum for the coupled and uncoupled case for L2/3 and L5/6

        # Time parameters
        dt = args.dt
        tstop = 600 # seconds
        t = np.arange(0, tstop, dt) # Note: np.arange excludes the stop so we add dt to include the last value
        transient = 10

        # Number of areas (intralaminar = 1)
        Nareas = 1

        # Load network config
        tau, sig, J_conn, Iext_conn, Ibgk_conn = get_network_configuration('interlaminar_a', noconns=False)
        Nbin = 100 # Number of bins for the power spectrum

        # Calculate the rate
        rate_conn = interlaminar_simulation(
            args.analysis, t, dt, tstop, J_conn, tau, sig, 
            Iext_conn, Ibgk_conn, args.sigmaoverride, Nareas
        )
        pxx_coupled_l23_bin, fxx_coupled_l23_bin, pxx_coupled_l56_bin, fxx_coupled_l56_bin = calculate_interlaminar_power_spectrum(rate_conn, dt, transient, Nbin)

        # Run simulation when the two layers are uncoupled
        tau, sig, J_noconn, Iext_noconn, Ibgk_noconn = get_network_configuration('interlaminar_u', noconns=False)

        rate_noconn = interlaminar_simulation(
            args.analysis, t, dt, tstop, J_noconn, tau, sig, 
            Iext_noconn, Ibgk_conn, args.sigmaoverride, Nareas
        )
        pxx_uncoupled_l23_bin, fxx_uncoupled_l23_bin, pxx_uncoupled_l56_bin, fxx_uncoupled_l56_bin = calculate_interlaminar_power_spectrum(rate_noconn, dt, transient, Nbin)

        # Plot spectrogram
        plot_interlaminar_power_spectrum(
            fxx_uncoupled_l23_bin, fxx_coupled_l23_bin,
            pxx_uncoupled_l23_bin, pxx_coupled_l23_bin,
            fxx_uncoupled_l56_bin, fxx_coupled_l56_bin,
            pxx_uncoupled_l56_bin, pxx_coupled_l56_bin,
            args.analysis
        )

        # Plot spectrogram using neurodsp
        plot_power_spectrum_neurodsp(dt,rate_conn, rate_noconn, 'interlaminar')

        # Pickle the results rate over time
        # Transform the results so that they are saved in a dic (similar to NeuroML output)
        pyrate = {
            'L23_E_Py/conn': rate_conn[0, :, 0],
            'L23_I_Py/conn': rate_conn[1, :, 0],
            'L56_E_Py/conn': rate_conn[2, :, 0],
            'L56_I_Py/conn': rate_conn[3, :, 0],
            'L23_E_Py/unconn': rate_noconn[0, :, 0],
            'L23_I_Py/unconn': rate_noconn[1, :, 0],
            'L56_E_Py/unconn': rate_noconn[2, :, 0],
            'L56_I_Py/unconn': rate_noconn[3, :, 0],
            'ts': t
        }

        picklename = os.path.join(args.analysis, 'simulation.pckl')
        if not os.path.exists(picklename):
            os.mkdir(os.path.dirname(picklename))

        with open(picklename, 'wb') as filename:
            pickle.dump(pyrate, filename)

        print('    Done Analysis!')

    if args.analysis == 'interlaminar_b':
        print('-----------------------')
        print('Interlaminar Simulation')
        print('-----------------------')

        # Calculates the spectogram and 30 traces of actvity in layer 5/6

        # Time parameters
        dt = args.dt
        tstop = 6000 # seconds
        transient = 10
        t = np.arange(dt+transient, tstop + dt, dt) # Note: np.arange excludes the stop so we add dt to include the last value
        
        # Number of areas (intralaminar = 1)
        Nareas = 1

        # Load network config
        tau, sig, J, Iext, Ibgk = get_network_configuration('interlaminar_b', noconns=False)

        # Frequencies of interest
        min_freq5 = 4 # alpha range
        min_freq2 = 30 # gama range

        # check if file with simulation exists, if not calculate the simulation
        simulation_file = os.path.join(args.analysis, 'simulation.pckl')
        if not os.path.isfile(simulation_file):
            print('    Re-calculating the simulation')
            rate = interlaminar_simulation(args.analysis, t, dt, tstop, J, tau, sig, Iext, Ibgk, args.sigmaoverride, Nareas)
        else:
            print('    Loading the pre-saved simulation file: %s' %simulation_file)
            with open(simulation_file, 'rb') as filename:
                rate = pickle.load(filename)

        # Analyse and Plot traces of activity in layer 5/6
        segment5, segment2, segindex, numberofzones = interlaminar_activity_analysis(rate, transient, dt, t, min_freq5)
        plot_activity_traces(dt, segment5, segindex, args.analysis)

        # # Analyse and Plot spectrogram of layer L2/3
        # ff, tt, Sxx = interlaminar_analysis_periodeogram(rate, segment2, transient, dt, min_freq2, numberofzones)

    if args.analysis == 'interlaminar_c':
        print('-----------------------')
        print('Interlaminar Figure 3C')
        print('-----------------------')

        dt = args.dt
        tstop = 600
        transient = 10
        Nareas = 1
        t = np.arange(0, tstop, dt)

        # Get network configuration for interlaminar sweep
        tau, sig, J, Iext_base, Ibgk = get_network_configuration('interlaminar_c', noconns=False)

        nruns = 10
        sweep_I_L56E = [3, 6, 9, 12]   # values from the paper

        results = interlaminar_3C_analysis(
            t, dt, tstop,
            J, tau, sig,
            Iext_base, Ibgk,
            args.sigmaoverride,
            Nareas,
            sweep_I_L56E,
            nruns,
            transient,
        )

        interlaminar_3C_plot(results, analysis_name=args.analysis)
        print('    Saved: interlaminar_c/interlaminar_3C.png')

    if args.analysis == 'interareal':
        print('-----------------------')
        print('Interareal Analysis (Figure 4)')
        print('-----------------------')

        # Time parameters
        dt = args.dt
        tstop = 40 # seconds
        transient = 5
        t = np.arange(0, tstop, dt)

        # Number of areas
        Nareas = 2
        areas = ["V1", "V4"]

        # Statistical runs
        nstats = 10

        # Frequency ranges
        minfreq_l23 = 30  # Gamma
        minfreq_l56 = 4   # Alpha

        # Load network config
        tau, sig, J, Iext, Ibgk = get_network_configuration('interareal', noconns=False)

        # Interareal connectivity
        W = build_interareal_W(Nareas)

        # Stimulus intensity as per paper: I=15
        stim_intensity = 15

        # ============================================
        # Stimulate V1, Record V4
        # ============================================
        print('    Stimulate V1, Record V4 (Fig 4A-C)')

        Iexts_V1 = np.array([2, 0, 4, 0])
        
        rate_rest_v1, rate_stim_v1 = interareal_simulation(
            t, dt, tstop,
            J, W, tau,
            Iexts_V1, Ibgk, sig,
            args.sigmaoverride,
            nstats=nstats,
            stim_area='V1',
            stim_intensity=stim_intensity
        )

        px20_v1, px2_v1, px50_v1, px5_v1, fx2_v1, pgamma_v1, palpha_v1 = interareal_analysis(
            rate_rest_v1, rate_stim_v1,
            transient, dt,
            minfreq_l23=minfreq_l23,
            minfreq_l56=minfreq_l56,
            nareas=Nareas,
            stats=nstats
        )

        interareal_plt(
            areas=areas,
            px20=px20_v1, px2=px2_v1,
            px50=px50_v1, px5=px5_v1,
            fx2=fx2_v1,
            stimulated_area='stimulate_V1',
            nstats=nstats
        )
        print(f'    Gamma p-value: {pgamma_v1:.4f}, Alpha p-value: {palpha_v1:.4f}')
        print('    Saved: interareal/stimulate_V1_*.png')

        # ============================================
        # Stimulate V4, Record V1
        # ============================================
        print('    Stimulate V4, Record V1 (Fig 4D-F)')

        Iexts_V4 = np.array([1, 0, 1, 0])
        
        rate_rest_v4, rate_stim_v4 = interareal_simulation(
            t, dt, tstop,
            J, W, tau,
            Iexts_V4, Ibgk, sig,
            args.sigmaoverride,
            nstats=nstats,
            stim_area='V4',
            stim_intensity=stim_intensity
        )

        px20_v4, px2_v4, px50_v4, px5_v4, fx2_v4, pgamma_v4, palpha_v4 = interareal_analysis(
            rate_rest_v4, rate_stim_v4,
            transient, dt,
            minfreq_l23=minfreq_l23,
            minfreq_l56=minfreq_l56,
            nareas=Nareas,
            stats=nstats
        )

        interareal_plt(
            areas=areas,
            px20=px20_v4, px2=px2_v4,
            px50=px50_v4, px5=px5_v4,
            fx2=fx2_v4,
            stimulated_area='stimulate_V4',
            nstats=nstats
        )
        print(f'    Gamma p-value: {pgamma_v4:.4f}, Alpha p-value: {palpha_v4:.4f}')
        print('    Saved: interareal/stimulate_V4_*.png')
        
        print('Interareal analysis completed.')

    if args.analysis == 'debug':
        print('-----------------------')
        print('Debugging')
        print('-----------------------')

        # Call a function that plots and saves of the firing rate for the intra- and interlaminar simulation
        print('Running debug simulation/analysis with %s'%args)

        dt = args.dt
        firing_rate_analysis(args.noconns, args.testduration, args.sigmaoverride, args.initialrate, dt)
    
    if not args.nogui:
        plt.show()