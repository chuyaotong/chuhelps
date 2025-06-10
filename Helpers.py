import cirq
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy as sp
from slab.datamanagement import SlabFile
import scipy.stats as st
import json
import textwrap
import inspect
from scipy.optimize import curve_fit
import math
from lmfit import Model
import time

class SQBSWAP(cirq.Gate):
    def __init__(self):
        super(SQBSWAP, self)

    def _num_qubits_(self):
        return 2

    def _unitary_(self):
        return np.array([[1/np.sqrt(2),0,0,-1j/np.sqrt(2)],
                          [0,1,0,0],
                          [0,0,1,0],
                          [-1j/np.sqrt(2),0,0,1/np.sqrt(2)]])

    def _circuit_diagram_info_(self, args):
        return "G"

def get_cirq(qubit, moment):
    if moment == "I":
        return cirq.I(qubit)
    elif moment == "X":
        return cirq.X(qubit)
    elif moment == "Y":
        return cirq.Y(qubit)
    elif moment == "Z":
        return cirq.Z(qubit)
    elif moment == "X/2":
        return cirq.rx(rads= np.pi/2)(qubit)
    elif moment == "-X/2":
        return cirq.rx(rads=-np.pi/2)(qubit)
    elif moment == "Y/2":
        return cirq.ry(rads= np.pi/2)(qubit)
    elif moment == "-Y/2":
        return cirq.ry(rads=-np.pi/2)(qubit)
    elif moment == "Z/2":
        return cirq.rz(rads= np.pi/2)(qubit)
    elif moment == "-Z/2":
        return cirq.rz(rads=-np.pi/2)(qubit)
    elif moment == "Z/4":
        return cirq.rz(rads= np.pi/4)(qubit)
    elif moment == "-Z/4":
        return cirq.rz(rads=-np.pi/4)(qubit)

def moments_to_cirq(momentList):
    q1, q2 = cirq.LineQubit.range(2)
    prep_circuit = cirq.Circuit()
    for i in momentList:
        prep_circuit.append(get_cirq(q1, i["Q1"]))
        prep_circuit.append(get_cirq(q2, i["Q2"]))
        if i["C"] == "ISWAP":
            prep_circuit.append(cirq.ISWAP(q1, q2))
        elif i["C"] == "SQISWAP":
            prep_circuit.append(cirq.SQRT_ISWAP(q1, q2))
        elif i["C"] == "SQBSWAP":
            prep_circuit.append(SQBSWAP()(q1, q2))
    return prep_circuit

def gauss(x, mu, sigma, A, C):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + C

def exp_sin(x, a, b, f, c, d):
    return a * np.exp(-x / b) * np.sin(2 * np.pi * f * x + c) + d

def damped_sin(x, A, tau, freq, phase, c):
    return (
        A * np.exp(-x / tau) 
        * np.sin(2 * np.pi * freq * x + phase) 
        + c)

def exp_decay(x, A, tau, c):
    return A * np.exp(-x / tau) + c

def exp_decay_offset(x, a, b, c):
    return a * np.exp(-x / b) + c

def bimodal(x, mu1, sigma1, A1, C1, mu2, sigma2, A2, C2):
    return gauss(x, mu1, sigma1, A1, C1) + gauss(x, mu2, sigma2, A2, C2)

def bimodal2(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x, mu1, sigma1, A1, 0) + gauss(x, mu2, sigma2, A2, 0)

def fanoresonance(x, f, kappa, q, A):
   """Fano resonance function"""
   return A*(q*kappa/2+x-f)**2/((kappa/2)**2+(x-f)**2)

def bg_offset_o1(x, x0, offset, slope):
    """Constant background with a sloping offset of order 1"""
    return offset + slope*(x - x0)

def fano_bg_o1(x, f0, kappa, q, A, offset, slope):
    """Fano resonance function with a sloping offset order 1"""
    fano = fanoresonance(x, f0, kappa, q, A)
    bg = bg_offset_o1(x, f0, offset, slope)
    return fano + bg

def double_fano_resonance(x, f, kappa, q, A, chi):
    """Double Fano resonance function for qubit in thermal state"""
    return fanoresonance(x, f - chi, kappa, q, A) + fanoresonance(x, f + chi, kappa, q, A)

def double_fano_bg_o1(x, f, kappa, q, A, chi, offset, slope):
    """Double Fano resonance function for qubit in thermal state"""
    double_fano = double_fano_resonance(x, f, kappa, q, A, chi)
    bg = bg_offset_o1(x, f, offset, slope)
    return double_fano + bg

def full_fname_to_data(full_fname):
    """
    Convert slab file to data that can be imported in Experiment.display()
    """
    with SlabFile(full_fname) as a:
        data = {}
        for k in a.keys():
            data[k] = np.array(a[k])
    with SlabFile(full_fname[:-3] + "_Config.h5") as a:
        cfg = {}
        for k in a.keys():
            cfg[k] = np.array(a[k])
    return {'config': cfg, 'data': data}

def flux_to_volt(flux, flux_period, flux_offset):
    return flux_period * flux + flux_offset

def volt_to_flux(volt, flux_period, flux_offset):
    return (volt - flux_offset) / flux_period  

def fluxon_freq_at_flux(flux):
    if flux > 0.5:
        return -3000.37 + 5938.44 * flux + 62
    else:
        return 3000.37 - 5938.44 * flux

def get_slope(di, dq, is_plot=False):
    # Use a K-Means Cluster to automatically assign points in IQ space, with 2 total clusters
    clf = KMeans(n_clusters=2, n_init=10)
    data = np.array(list(zip(di, dq)))
    clf.fit(data)
    # Find the centroids of each cluster
    centers = clf.cluster_centers_
    labels = clf.predict(data)

    # Determine the perpendicular slope to the line that connects the two centroids
    slope = ((centers[1][1] - centers[0][1]) / (centers[1][0] - centers[0][0]))
    angle = - np.arctan2((centers[1][1] - centers[0][1]), (centers[1][0] - centers[0][0]))

    if angle < 0:
        angle = angle + np.pi

    if np.abs(slope) > 0.1:
        slope = -1 / slope

    # Find the midpoint of the two centroids.
    if centers[1][0] > centers[0][0]:
        midpointX = (centers[1][0] - centers[0][0]) / 2
    else:
        midpointX = (centers[0][0] - centers[1][0]) / 2

    if centers[1][1] > centers[0][1]:
        midpointY = (centers[1][1] - centers[0][1]) / 2
    else:
        midpointY = (centers[0][1] - centers[1][1]) / 2

    # Find the line that goes through the midpoint of the centroids and has the perpendicular slope
    intercept = midpointY - slope * midpointX

    if is_plot:
        plt.close()
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, title="IQ Readout", xlabel='di', ylabel='dq')
        ax.scatter(di, dq, s=5)  # , c = labels
        ax.scatter(centers[:, 0], centers[:, 1])
        plt.show()

    return slope, intercept, angle

def histogram_fit(di, dq, slope, intercept, plot_histogram=False):
    signal = dq * slope + di - intercept

    if plot_histogram:
        plt.close()
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, title="Single Shot Histogram", xlabel='signal', ylabel='frequency')
        n, bins, patches = ax.hist(signal, bins=80)
    else:
        n, bins = np.histogram(signal, bins=80)

    amp = np.max(n)
    peak = bins[np.argmax(n)]
    x = (bins[1:] + bins[:-1]) / 2

    if peak < bins[40]:
        peak1 = peak
        #CD: Fixed peak finding error 2/16/25
        peak2 = bins[np.argmax(n[40:])+40]
    else:
        peak1 = bins[np.argmax(n[:40])]
        peak2 = peak

    expected = (peak1, 20, amp, peak2, 20, amp)
    try:
        #CD: Fixed bounds error 2/16/25
        params, cov = sp.optimize.curve_fit(bimodal2, x, n, expected, bounds = ((x[0], 0, 0, x[0], 0, 0), (x[-1], 1000, 1.5*amp, x[-1], 1000, 1.5*amp)))
    except RuntimeError:
        print("Bad fit")
        params = expected

    params = list(params)
    params[1] = abs(params[1])
    params[4] = abs(params[4])

    newX = np.linspace(x[0], x[-1], 1000)
    newY = bimodal2(newX, *params)

    peak1 = np.where(newX > params[0])[0][0]
    peak2 = np.where(newX > params[3])[0][0]
    if peak1 > peak2:
        peak1, peak2 = peak2, peak1

    ymin = np.argmin(newY[peak1:peak2])

    demarcation_line = newX[peak1 + ymin]

    if plot_histogram:
        ax.plot(newX, newY)
        ax.axvline(demarcation_line)
        plt.show()

    return n, bins, params, demarcation_line

def find_first_peak(xdata, ydata):
    current_peak_loc = np.argmax(ydata)
    previous_peak_loc = current_peak_loc 
    #Loop until the previous peak is within 2 points of the current peak
    while np.argmax(ydata[:current_peak_loc]) - previous_peak_loc > 2:
        previous_peak_loc = current_peak_loc
        current_peak_loc = np.argmax(ydata[:current_peak_loc])
    peak_val = np.round(xdata[int(current_peak_loc)], 2)
    return(peak_val, current_peak_loc)

def stat_infidelity(params, demarcation_line):
    
    stat_err_g = st.norm.cdf( (demarcation_line - params[0]) / params[1] )
    stat_err_e = st.norm.cdf( (demarcation_line - params[3]) / params[4] )
    if stat_err_g < 0.5:
        stat_err_g = 1 - stat_err_g
    if stat_err_e < 0.5:
        stat_err_e = 1 - stat_err_e

    avg_stat_err = np.min([stat_err_g, stat_err_e])
    return(avg_stat_err)

def postselect(di1, demarcation_line):
    loc_g = np.where((di1[0] < demarcation_line))  # 1st readout g
    loc_e = np.where((di1[0] > demarcation_line))  # 1st readout e

    loc_gg = np.where((di1[1][loc_g] < demarcation_line))  # 2nd readout gg
    loc_ge = np.where((di1[1][loc_g] > demarcation_line))  # 2nd readout ge

    loc_eg = np.where((di1[1][loc_e] < demarcation_line))  # 2nd readout eg
    loc_ee = np.where((di1[1][loc_e] > demarcation_line))  # 2nd readout ee

    p_gg = len(loc_gg[0]) / len(loc_g[0])
    p_ge = len(loc_ge[0]) / len(loc_g[0])
    p_eg = len(loc_eg[0]) / len(loc_e[0])
    p_ee = len(loc_ee[0]) / len(loc_e[0])
    return([p_gg, p_ge, p_eg, p_ee])

def slice_and_normalize_ms(iqlist, ro_clicks = None, ro_clicks_m1 = None, ro_clicks_m2 = None):
    '''
    Takes IQ data and slices it into two arrays (i and q) for two readouts (m1 and m2).
    '''         
    iqlist = iqlist.astype(float)

    di = iqlist[...,0]
    dq = iqlist[...,1]

    if ro_clicks is not None:
        di /= ro_clicks
        dq /= ro_clicks

    m1_di = di[:, 0]
    m1_dq = dq[:, 0]
    m2_di = di[:, 1]
    m2_dq = dq[:, 1]

    if ro_clicks is None and ro_clicks_m1 is not None and ro_clicks_m2 is not None:
        m1_di /= ro_clicks_m1
        m1_dq /= ro_clicks_m1
        m2_di /= ro_clicks_m2
        m2_dq /= ro_clicks_m2

    return (m1_di, m1_dq, m2_di, m2_dq)

def select(m1_di, m1_dq, m2_di, m2_dq, 
           demarcation_line, auto_cali = False,
            verbose = True, silence = False):
    #TODO use gaussian fit instead of demarcation lines for selections 
    
    if isinstance(demarcation_line, (int, float)):
        demarcation_line = [demarcation_line, demarcation_line]
    elif not (isinstance(demarcation_line, list) and len(demarcation_line) == 2):
        print(f"I don't understand your demarcation line {demarcation_line}")   

    if auto_cali:

        # get rotation angle from m1
        slope, intercept, angle = get_slope(m1_di, m1_dq, is_plot = False)
        # rotate m1 and m2 data
        angle = (angle - np.pi) % (2 * np.pi)
        angle_deg = angle * 180 / np.pi

        if verbose:
            print(f"auto rotating both M1 and M2 by {angle_deg} deg")

        rotated_m1_di, rotated_m1_dq = rotate_iq(m1_di, m1_dq, angle)
        rotated_m2_di, rotated_m2_dq = rotate_iq(m2_di, m2_dq, angle)

        params = fit_double_gaussian(rotated_m1_di, fix_C = 0, silence = silence)
        demarcation_line = [min(params[0],params[3]), max(params[0],params[3])]

        if verbose:
            print(f"overwriting demarcation line to {demarcation_line}")

        m1_di = rotated_m1_di
        m1_dq = rotated_m1_dq
        m2_di = rotated_m2_di
        m2_dq = rotated_m2_dq

    else:
        angle_deg = 0

    # Assuming rotation in phase already applied so all info on di

    m1_g = np.where((m1_di < demarcation_line[0]))[0]  # index list of 1st readout g
    m1_e = np.where((m1_di > demarcation_line[1]))[0]  # index list of 1st readout e

    m2_di_m1g = np.array(m2_di[m1_g])
    m2_dq_m1g = np.array(m2_dq[m1_g])

    m2_di_m1e = np.array(m2_di[m1_e])
    m2_dq_m1e = np.array(m2_dq[m1_e])

    return m2_di_m1g, m2_dq_m1g, m2_di_m1e, m2_dq_m1e, demarcation_line, angle_deg

def postselect_tprocv2(di, dq=None, demarcation_line=None):
    all_probs = []
    # reps, sweep, readout trigger
    # num_sweeps = di.shape[0] # each individual sweep

    # TOCHECK why use demarcation lines type for this?
    if type(demarcation_line) is not list:
        if demarcation_line is None:
            # Automatically find the demarcation line based on the di/dq of the first measurement in each sweep param
            n, bins_a, params_a, demarcation_line = histogram_fit(di[:, 0, 0], dq[:, 0, 0], slope=0, intercept=0, plot_histogram = False)
        num_sweeps = di.shape[1]
        print(num_sweeps)
        print(di.shape[0])
        print(di.shape[1])
        print(di.shape)
    else:
        num_sweeps = di.shape[0]

    for i in range(num_sweeps):
        if type(demarcation_line) is list:
            curr_demarcation_line = demarcation_line[i]
            first_meas = di[i, :, 0]
            second_meas = di[i, :, 1]
        else:
            curr_demarcation_line = demarcation_line
            first_meas = di[:, i, 0]
            second_meas = di[:, i, 1]

        loc_g = np.where((first_meas < curr_demarcation_line))  # 1st readout g
        loc_e = np.where((first_meas > curr_demarcation_line))  # 1st readout e

        if len(loc_g[0])/len(first_meas) < 0.1:
            print(f'Not enough shots in g in sweep {i}')
            all_probs.append([0, 0, 0, 0])
        elif len(loc_e[0])/len(first_meas) < 0.1:
            print(f'Not enough shots in e in sweep {i}')
            all_probs.append([0, 0, 0, 0])

        else:
            loc_gg = np.where((second_meas[loc_g] < curr_demarcation_line))  # 2nd readout gg
            loc_ge = np.where((second_meas[loc_g] > curr_demarcation_line))  # 2nd readout ge
            loc_eg = np.where((second_meas[loc_e] < curr_demarcation_line))  # 2nd readout eg
            loc_ee = np.where((second_meas[loc_e] > curr_demarcation_line))  # 2nd readout ee

            p_gg = len(loc_gg[0]) / len(loc_g[0])
            p_ge = len(loc_ge[0]) / len(loc_g[0])
            p_eg = len(loc_eg[0]) / len(loc_e[0])
            p_ee = len(loc_ee[0]) / len(loc_e[0])
            all_probs.append([p_gg, p_ge, p_eg, p_ee])
    return(all_probs)

def postselect_tprocv2_M2diff(di, dq=None, demarcation_line=None):
    all_probs = []
    # reps, sweep, readout trigger

    num_sweeps = di.shape[0]

    for i in range(num_sweeps):
        first_meas = di[i, :, 0]
        second_meas = di[i, :, 1]
        dline = demarcation_line[i]

        loc_g = np.where((first_meas < dline[0]))  # 1st readout g
        loc_e = np.where((first_meas > dline[0]))  # 1st readout e

        if len(loc_g[0])/len(first_meas) < 0.1:
            print(f'Not enough shots in g in sweep {i}')
            all_probs.append([0, 0, 0, 0])
        elif len(loc_e[0])/len(first_meas) < 0.1:
            print(f'Not enough shots in e in sweep {i}')
            all_probs.append([0, 0, 0, 0])

        else:
            loc_gg = np.where((second_meas[loc_g] < dline[1]))  # 2nd readout gg
            loc_ge = np.where((second_meas[loc_g] > dline[1]))  # 2nd readout ge
            loc_eg = np.where((second_meas[loc_e] < dline[1]))  # 2nd readout eg
            loc_ee = np.where((second_meas[loc_e] > dline[1]))  # 2nd readout ee

            p_gg = len(loc_gg[0]) / len(loc_g[0])
            p_ge = len(loc_ge[0]) / len(loc_g[0])
            p_eg = len(loc_eg[0]) / len(loc_e[0])
            p_ee = len(loc_ee[0]) / len(loc_e[0])
            all_probs.append([p_gg, p_ge, p_eg, p_ee])
    return(all_probs)

def postselect_tprocv2_CT(di, dq=None, demarcation_line=None, get_probs = False, get_m2 = False, rotate_iq = None):
    all_probs = []
    # reps, sweep, readout trigger
    # num_sweeps = di.shape[0] # each individual sweep

    # TOCHECK why use demarcation lines type for this?
    #CD - this was originally intended if demarcation_line was not initiated, as I set the default to be None. I 
    # didn't want to set it to be zero, since that could be a valid demarcation line. In addition, for expts where
    # I was sweeping through many different readout gains, the demarcation line would change every time. At one point, 
    # I was hoping to find those demarcation lines manually, especially when I wasn't confident that histogram_fit 
    # would give me a good result each time. 4/13/2025
    if demarcation_line is None:
        # Automatically find the demarcation line based on the di/dq of the first measurement in each sweep param
        n, bins_a, params_a, demarcation_line = histogram_fit(di[:, 0, 0], dq[:, 0, 0], slope=0, intercept=0, plot_histogram = False)
        demarcation_line = [demarcation_line, demarcation_line]
    elif isinstance(demarcation_line, (int, float)):
        demarcation_line = [demarcation_line, demarcation_line]
    elif not (isinstance(demarcation_line, list) and len(demarcation_line) == 2):
        print_location()
        print("I don't understand your demarcation line, maybe you want to use the old function")   

    num_sweeps = di.shape[1]

    for i in range(num_sweeps):
        di_m1 = di[:, i, 0]
        di_m2 = di[:, i, 1]

        idx_m1_g = np.where((di_m1 < demarcation_line[0]))  # 1st readout g
        idx_m1_e = np.where((di_m1 > demarcation_line[1]))  # 1st readout e
            
        if len(idx_m1_g[0])/len(di_m1) < 0.05:
            print(f'Less than 5% shots in g during M1 in sweep {i}')
            all_probs.append([0, 0, 0, 0])
        elif len(idx_m1_e[0])/len(di_m1) < 0.05:
            print(f'Less than 5% shots in e during M1 in sweep {i}')
            all_probs.append([0, 0, 0, 0])
        else:
            idx_gg = np.where((di_m2[idx_m1_g] < demarcation_line[0]))
            idx_ge = np.where((di_m2[idx_m1_g] > demarcation_line[1]))  
            idx_eg = np.where((di_m2[idx_m1_e] < demarcation_line[0]))  
            idx_ee = np.where((di_m2[idx_m1_e] > demarcation_line[1]))  

            p_gg = len(idx_gg[0]) / (len(idx_gg[0])+len(idx_ge[0])) # m2 is g given m1 is g
            p_ge = len(idx_ge[0]) / (len(idx_gg[0])+len(idx_ge[0])) # m2 is e given m1 is g
            p_eg = len(idx_eg[0]) / (len(idx_eg[0])+len(idx_ee[0])) # m2 is g given m1 is e
            p_ee = len(idx_ee[0]) / (len(idx_eg[0])+len(idx_ee[0])) # m2 is e given m1 is e
            all_probs.append([p_gg, p_ge, p_eg, p_ee])
    
    return (all_probs)
    # TODO combine select into this
        # elif get_m2:
        #     m2_di_m1g = np.array(di[idx_m1_g, 1])
        #     m2_dq_m1g = np.array(dq[idx_m1_g, 1])

        #     m2_di_m1e = np.array(di[idx_m1_e, 1])
        #     m2_dq_m1e = np.array(dq[idx_m1_e, 1])
        #     return m2_di_m1g, m2_dq_m1g, m2_di_m1e, m2_dq_m1e

        # else:
        #     print('Hi there, I selected, but you didnt ask me to do anything...')

def postselect_res_tprocv2(di, dq, demarcation_line=None, getRatio = False):
    all_ratios = []
    # reps, sweep, readout trigger
    num_sweeps = di.shape[1] # each individual sweep

    if demarcation_line is None:
        demarcation_line = np.average(di[:, 0, 0])

    for i in range(num_sweeps):
        first_meas_di = di[:, i, 0]
        second_meas_di = di[:, i, 1]
        first_meas_dq = dq[:, i, 0]
        second_meas_dq = dq[:, i, 1]

        loc_g = np.where((first_meas_di < demarcation_line))  # 1st readout g
        loc_e = np.where((first_meas_di > demarcation_line))  # 1st readout e

        if getRatio:
            print("Ratio 1g/1e: {0:.2f}".format(len(loc_g[0]) / len(loc_e[0])))

        resMeas1G = np.average(np.abs(second_meas_di[loc_g] * 1j + second_meas_dq[loc_g]))
        resMeas1E = np.average(np.abs(second_meas_di[loc_e] * 1j + second_meas_dq[loc_e]))
        all_ratios.append([resMeas1G, resMeas1E])

    return(all_ratios)

def postselectResMeas(di1, dq1, demarcation_line, getRatio = False):
    loc_g = np.where((di1[0] < demarcation_line))  # 1st readout g
    loc_e = np.where((di1[0] > demarcation_line))  # 1st readout e

    if getRatio:
        print("Ratio 1g/1e: {0:.2f}".format(len(loc_g[0]) / len(loc_e[0])))

    resMeas1G = np.average(np.abs(di1[1][loc_g] * 1j + dq1[1][loc_g]))
    resMeas1E = np.average(np.abs(di1[1][loc_e] * 1j + dq1[1][loc_e]))

    return([resMeas1G, resMeas1E])

def lorentzian(x, x0, gamma, A, C):
    return A * ( (gamma/2) / ( (x - x0) ** 2 + (gamma / 2) ** 2) ) + C

def full_fname_to_data(full_fname):
    """
    Convert slab file to data that can be imported in Experiment.display()
    """
    with SlabFile(full_fname) as a:
        data = {}
        for k in a.keys():
            data[k] = np.array(a[k])
    # with (full_fname[:-3] + "_cfg.json") as a:
    with open(full_fname[:-3] + "_cfg.json") as json_file:
        config = json.load(json_file)

    #     cfg = {}
    #     for k in a.keys():
    #         cfg[k] = np.array(a[k])
    # return {'config': cfg, 'data': data}

def postselectResMeas(di1, dq1, demarcation_line, getRatio = False):
    loc_g = np.where((di1[0] < demarcation_line))  # 1st readout g
    loc_e = np.where((di1[0] > demarcation_line))  # 1st readout e

    if getRatio:
        print("Ratio 1g/1e: {0:.2f}".format(len(loc_g[0]) / len(loc_e[0])))

    resMeas1G = np.average(np.abs(di1[1][loc_g] * 1j + dq1[1][loc_g]))
    resMeas1E = np.average(np.abs(di1[1][loc_e] * 1j + dq1[1][loc_e]))

    return([resMeas1G, resMeas1E])

def postselectFirstResMeas(di1, dq1, demarcation_line, getRatio = False):
    loc_g = np.less(di1[0], demarcation_line)  # 1st readout g
    loc_e = np.greater(di1[0], demarcation_line)  # 1st readout e

    if getRatio:
        print("Ratio 1g/1e: {0:.2f}".format(len(loc_g[0]) / len(loc_e[0])))

    # print(di1.shape)

    di_g = np.ma.masked_array(di1[1], mask = loc_g)
    di_e = np.ma.masked_array(di1[1], mask = loc_e)

    # print(di_g.shape)
    # print(di_e)

    dq_g = np.ma.masked_array(dq1[1], mask = loc_g)
    dq_e = np.ma.masked_array(dq1[1], mask = loc_e)

    resMeas1G = np.array(np.ma.mean(np.ma.MaskedArray.__abs__(di_g * 1j + dq_g),axis=1))
    resMeas1E = np.array(np.ma.mean(np.ma.MaskedArray.__abs__(di_e * 1j + dq_e),axis=1))

    # print(resMeas1E.shape)
    # print(resMeas1G.shape)

    return([resMeas1G, resMeas1E])

def reset_dc_volt_to_zero(soc, channels = [0, 1, 2, 3, 4, 5, 6], exp_flux_ch = None, verbose=False):
    reset_chs = [ch for ch in channels if ch != exp_flux_ch]
    if verbose:
        print(f'Setting DC bias channel {reset_chs} to 0 V')
    for ch in reset_chs:
        soc.rfb_set_bias(ch, 0)  #channel, voltage

def iqdata(iqlist):
    # I quadrature data
    di = iqlist[:,0]

    # Q quadrature data
    dq = iqlist[:,1]

    # Compute amplitude and phase from the I and Q quadratures
    amp = np.abs(iqlist.dot([1, 1j]))
    phase = (np.arctan(dq/di))

    return di, dq, amp, phase

def make_plot_title(config, dfilename, addition = ''):
    if 'display_cfg' not in config:
        return dfilename + ' ' + config['qubit_label']
    else:
        if all(k in config['display_cfg'] for k in ('res_gain', 'res_att_1', 'res_att_2')):
            config['res_power [dB]'] = all_power_to_db(
                config['res_att_1'], config['res_att_2'], config['res_gain'])
            res_gain_idx = config['display_cfg'].index('res_gain')
            config['display_cfg'].insert(res_gain_idx, 'res_power [dB]')

        if all(k in config['display_cfg'] for k in ('pulse_gain', 'pulse_att_1', 'pulse_att_2')):
            config['pulse_power [dB]'] = all_power_to_db(
                config['pulse_att_1'], config['pulse_att_2'], config['pulse_gain'])
            pulse_gain_idx = config['display_cfg'].index('pulse_gain')
            config['display_cfg'].insert(pulse_gain_idx, 'pulse_power [dB]')

        displayed_keys = ", ".join(
            f"{key}: {round_if_number(config.get(key, 'N/A'))}"
            for key in config['display_cfg']
            )

        wrapped_keys = "\n".join(textwrap.wrap(displayed_keys, width=60))
        title_text = (
                dfilename + ' ' + addition + config['qubit_label'] + 
                '\n' + wrapped_keys
                )
        return title_text

def print_location():
    frame = inspect.currentframe().f_back
    func_name = frame.f_code.co_name
    class_name = frame.f_locals.get('self', None).__class__.__name__ if 'self' in frame.f_locals else None

    print(f"Called from function: {func_name}, class: {class_name}")

def quick_plot_2d(xdata, ydata, zdata, datapath, dfilename,
            title_text = None, xlabel = ' ', ylabel = ' ', vmin = None, vmax = None):
    plt.close()

    if title_text is None:
        title_text = dfilename

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(111, title = title_text,
                            xlabel = xlabel,
                            ylabel = ylabel)
    im = ax.pcolormesh(xdata, ydata, zdata, cmap = 'viridis', shading = 'nearest')

    if vmin is not None and vmax is not None:
        im.set_clim(vmin=vmin, vmax=vmax)

    fig.colorbar(im, ax=ax)
    plt.savefig(datapath + '/png/' + dfilename + '.png')
    plt.show()

def fit_gaussian(data, verbose = False, fix_C = None, silence = False):
    counts, bins = np.histogram(data, bins=80)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    model = Model(gauss)
    params = model.make_params(
        mu=np.mean(data), #mu
        sigma=np.std(data), #sigma
        A=np.max(counts), #amplitude
        C=data[0] #bg constant
    )

    if fix_C is None:
        params['C'].set(vary=True)
    elif isinstance(fix_C, (float, int)):
        params['C'].set(value=fix_C, vary=False)
    else:
        raise TypeError("fix_C must be a float or int")

    result = model.fit(counts, params, x=bin_centers)
    mu    = result.params['mu'].value
    sigma = result.params['sigma'].value
    A     = result.params['A'].value
    C     = result.params['C'].value
    fit_results = [mu, sigma, A, C]

    if not verbose and not silence:
        print(f"mu = {mu:.2f}, sigma = {sigma:.2f}, A = {A:.2f}, C = {C:.2f}")
    elif verbose:
        print(result.fit_report())

    return fit_results

def fit_double_gaussian(data, verbose = False, fix_C = None, silence = False):
    counts, bins = np.histogram(data, bins=80)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    model = Model(bimodal2)
    params = model.make_params(
        mu1 = dict(value = bin_centers[30], min = min(data), max = max(data)),
        sigma1 = dict(value=abs(bin_centers[40]-bin_centers[20])/2,  min = 0),
        A1     = dict(value=counts[30],                              min = 0),
        mu2 =   dict(value = bin_centers[50], min = min(data), max = max(data)),
        sigma2 = dict(value=abs(bin_centers[40]-bin_centers[20])/2,  min = 0),
        A2     = dict(value=counts[50],                              min = 0),
    )

    result = model.fit(counts, params, x=bin_centers)
    mu1    = result.params['mu1'].value
    sigma1 = result.params['sigma1'].value
    A1     = result.params['A1'].value
    mu2    = result.params['mu2'].value
    sigma2 = result.params['sigma2'].value
    A2     = result.params['A2'].value
    # C     = result.params['C'].value
    fit_results = [mu1, sigma1, A1, mu2, sigma2, A2] #, C]

    if not verbose and not silence:
        print(f"mu1 = {mu1:.2f}, sigma1 = {sigma1:.2f}, A1 = {A1:.2f}, \n mu2 = {mu2:.2f}, sigma2 = {sigma2:.2f}, A2 = {A2:.2f}")
    elif verbose:
        print(result.fit_report())

    return fit_results

def all_power_to_db(att_1, att_2, gain):
    return - (att_1 + att_2 - 20*math.log10(gain))

def fit_lorentzian(xdata, ydata, return_std = False, verbose=False):

    peak_idx_guess = np.argmax(np.abs(ydata - np.mean(ydata)))

    x0_guess = xdata[peak_idx_guess]
    A_guess = ydata[peak_idx_guess] - np.mean(ydata)
    C_guess = np.mean(ydata)
    guess = [x0_guess, 1, A_guess, C_guess]

    # lower = [-np.inf, 0,   0, 0]   # gamma ≥ 0
    # upper = [np.inf, np.inf, np.inf,  np.inf]
    # bounds = (lower, upper)
    params, cov = curve_fit(lorentzian, xdata, ydata, p0 = guess, maxfev=10000)
    param_stds = np.sqrt(np.diag(cov))

    if verbose:
        print(f'x0 = {params[0]:.4f}, gamma = {params[1]:.2f}, A = {params[2]:.2f}, C = {params[3]:.2f}')
    
    if return_std:
        return params, param_stds
    else:
        return params

def generic_fit(model_func, xdata, ydata, init_params, weights = None, 
                verbose=False, silence = False, return_std=False):
    """
    Generic function to perform a curve fitting with lmfit.Model.

    Parameters:
      model_func: callable
          The model function to fit.
      xdata, ydata: array_like
          Data to be fitted.
      init_params: dict
          Dictionary with initial parameter guesses.
      verbose: bool, optional
          Whether to print the detailed fit report.
      return_std: bool, optional
          Whether to return parameter uncertainties along with values.

    Returns:
      param_values: list
          Fitted parameter values.
      param_stds: list (if return_std is True)
          Uncertainties of the fitted parameters.
    """
    model = Model(model_func)

    params = model.make_params(**init_params)
    
    if weights is None:
        result = model.fit(ydata, params, x = xdata)
    else:
        result = model.fit(ydata, params, x = xdata, weights = weights)
    
    param_values = [param.value for param in result.params.values()]
    param_stds = [param.stderr for param in result.params.values()]
    if verbose:
        print(result.fit_report())
    elif not silence:
        out_str = ""
        for key, par in result.params.items():
            std = par.stderr if par.stderr is not None else 0.
            out_str += f"{key} = {par.value:.4f} ± {std:.4f}   "
        print(out_str)
    
    if return_std:
        return param_values, param_stds
    else:
        return param_values

def set_bias(config,
             soc = None, yoko = None, 
             verbose = False):
    '''
    I'm going back to a philosophy of being explicit about what kind of bias setting mode you want
    in order to prevent mistakes.
    Valid modes are:
    'yoko_volt', 'yoko_volt_phi0', 
    'yoko_curr', 'yoko_curr_phi0', 
    'soc_volt', 'soc_volt_phi0'
    If you use one of the *_phi0 modes, you need to set the flux_in_phi0 and flux_period in the config file.
    This cannot hotswap between different flux calibrations at the moment.

    '''
    cfg_keys = config.keys()
    # Set DC flux bias
    if 'exp_flux_ch' in cfg_keys:
        exp_flux_ch = config['exp_flux_ch']
    else:
        exp_flux_ch = config['flux_ch']

    reset_dc_volt_to_zero(soc, exp_flux_ch = exp_flux_ch) 
    
    if 'bias_mode' not in cfg_keys:
        print('Valid bias modes are: yoko_volt, yoko_curr, soc_volt, ' \
        'yoko_volt_phi0, yoko_curr_phi0, soc_volt_phi0')
        raise ValueError("bias_mode not defined in config file")

    bias_mode = config['bias_mode']
    if bias_mode[-4:] == 'phi0':
        config['flux_bias'] = flux_to_volt(config['flux_in_phi0'], 
                                      config['flux_period'], 
                                      config['flux_offset'])
        if verbose:
            print("Converted phi0 of %f Phi_0 to %f" % (config['flux_in_phi0'], config['flux_bias']))
    
    if bias_mode[:4] == 'yoko':

        if yoko is None:
            raise ValueError("Yoko object is not defined, please provide it")

        if bias_mode[:9] == 'yoko_volt': 

            if config['flux_bias'] < 0:

                print('WARNING: Yoko voltage is negative, setting to 0V')
                config['flux_bias'] = 0

            elif config['flux_bias'] > 10: #TODO set this range to a dynamic toggle
                print('WARNING: Yoko voltage is higher than 10V, setting to 10V')
                config['flux_bias'] = 10

            if verbose:
                print(f'yoko {yoko} as flux voltage source {config['flux_bias']} V')
            
            yoko.set_volt(config['flux_bias'])
        
        elif bias_mode[:9] == 'yoko_curr':

            if config['flux_bias'] < 0:

                print('WARNING: Yoko current is negative, setting to 0A')
                config['flux_bias'] = 0

            elif config['flux_bias'] > 200e-3:

                print('WARNING: Yoko current is higher than 200mA, setting to 200mA')
                config['flux_bias'] = 200e-3
            
            if verbose:
                print(f'yoko {yoko} as flux current source {config['flux_bias']*1E3} mA')
            
            yoko.set_current(config['flux_bias'])
    
    elif bias_mode[:3] == 'soc':

        if soc is None:
            raise ValueError("rfsoc object is not defined, please provide it")
        
        if config['flux_bias'] < 0:

            print('WARNING: rfsoc voltage is negative, setting to 0V')
            config['flux_bias'] = 0

        elif config['flux_bias'] > 10:

            print('WARNING: rfsoc voltage is higher than 10V, setting to 10V')
            config['flux_bias'] = 10

        if verbose:
            print("Ch: %d, Flux: %f V" % (config['flux_ch'], config['flux_bias']))

        soc.rfb_set_bias(exp_flux_ch, config['flux_bias'])
    else:
        raise ValueError(f"bias_mode {bias_mode} not recognized, check your config file")
    
    if 'DC_settle_time' in cfg_keys:
        time.sleep(config['DC_settle_time']) #Voltage/current settling time
    else:
        print("No DC settle time defined in config, using default of 1s")
        time.sleep(1)

    #Returns config with auto adjusted values
    return config

def set_bias_label(config,
             soc = None, yoko = None):
    
    cfg_keys = config.keys()

    if 'bias_mode' not in cfg_keys:
        print('Valid bias modes are: yoko_volt, yoko_curr, soc_volt, ' \
        'yoko_volt_phi0, yoko_curr_phi0, soc_volt_phi0')
        raise ValueError("bias_mode not defined in config file")

    bias_mode = config['bias_mode']

    if bias_mode[-4:] == 'phi0':

        flux_label = r"Flux $[\Phi_0]$"
    
    else:

        if bias_mode[:4] == 'yoko':

            if bias_mode[:9] == 'yoko_volt': 

                flux_label = r"Flux $[V]$"  

            elif bias_mode[:9] == 'yoko_curr':
                
                flux_label = r"Flux $[A]$"
        
        elif bias_mode[:3] == 'soc':
            
            flux_label = r"Flux $[V]$"

        else:
            raise ValueError(f"bias_mode {bias_mode} not recognized, check your config file")
    
    return flux_label
    
def pad_list(cfg_list, qubit_number, total_qubits):
    """
    We do a rather silly way of storing data in cfg. This means that it's rather hard to pass a list
    on the fly. This function pads the desired list and places it into a larger list with empty values
    so that the cfg can be interpreted correctly. 
    """
    padded_list = [0] * total_qubits
    padded_list[qubit_number] = cfg_list
    return padded_list

def prune_config(config_path, which_qubit):

    #CT I changed this to include the load json bit. 
    #CD - thanks! I think there's a bug here - json_file should not be passed in, since you 
    # use open(config_path) to load the file. Can I go ahead and remove it? It will probably 
    # break your current usage of this. 
    with open(config_path) as json_file:
        config = json.load(json_file)

    config['which_qubit'] = which_qubit

    # Reduces the config from having lists to only having one value for each key
    cfg_keys = config.keys()
    if 'which_qubit' in cfg_keys:
        num_qubits = config['num_qubits']
        for key in cfg_keys:
            if (
                key != 'display_cfg' 
                and type(config[key]) == list 
                and len(config[key]) == num_qubits
                ):
                try:
                    config[key] = config[key][config['which_qubit']]
                except Exception as e:
                    print(f'Your config key lists {key} {config[key]} does not work!')
    return config

#TODO - create a new file with just helper fits?
def fit_damped_sin(xdata, ydata, verbose=False, return_std=False):

    if ydata[0] > np.mean(ydata):
        phase_guess = np.pi / 2
    else:
        phase_guess = -np.pi / 2

    init_params = {
        'A': dict(value = abs((np.max(ydata) - np.min(ydata)) / 2), min = 0, max = 1),  # amplitude
        'tau': dict(value = np.max(xdata) / 2, min = 0), # assuming ~2T2 decay time
        'freq': dict(value = 5 / np.max(xdata),  min = 0),# assuming ~5 periods in the data
        'phase': dict(value = phase_guess, min = -np.pi, max = np.pi),
        'c': dict(value = np.mean(ydata), min = 0)
    }
    return generic_fit(damped_sin, xdata, ydata, init_params, verbose=verbose, return_std=return_std)

def fit_exp_decay(xdata, ydata, verbose=False, return_std=False):

    init_params = {
        'A': dict(value = ydata[0], min = -1, max = 1),         # amplitude
        'tau': dict(value = np.max(xdata) / 2, min = 0),  # decay constant, assuming ~2T1
        'c': dict(value = np.mean(ydata), min = 0)       # offset
    }

    return generic_fit(exp_decay, xdata, ydata, init_params, verbose=verbose, return_std=return_std)

def fit_fano(xdata, ydata, verbose=False, return_std=False):

    init_params = {
        'f': np.mean(xdata),      # center frequency
        'A': np.max(ydata) - np.min(ydata),  # amplitude scaling
        'kappa': 1e-3,
        'q': 1                    # Fano asymmetry parameter
    }

    return generic_fit(fanoresonance, xdata, ydata, init_params, verbose=verbose, return_std=return_std)

def fit_fano_bg(xdata, ydata, verbose=False, return_std=False):

    init_params = {
        'f0': np.mean(xdata),      # center frequency
        'A': abs(ydata[-1] - ydata[0]),  # amplitude scaling
        'kappa': 1e-3,
        'q': 1,                    # Fano asymmetry parameter
        'offset': np.mean(ydata),
        'slope': (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])              # slope of the background
    }

    return generic_fit(fano_bg_o1, xdata, ydata, init_params, verbose=verbose, return_std = return_std)

def fit_double_fano(xdata, ydata, verbose=False, silence = False, return_std=False):
    init_params = {
        'f': np.mean(xdata),      # center frequency
        'A': np.max(ydata) - np.min(ydata),  # amplitude scaling
        'kappa': 1e-3,
        'q': 1,                    # Fano asymmetry parameter
        'chi' : 1e-3
    }

    if verbose:
        print(f'initial guesses: {init_params}')

    return generic_fit(double_fano_resonance, xdata, ydata, init_params, verbose=verbose, 
                       silence=silence, return_std = return_std)

def fit_double_fano_bg(xdata, ydata, verbose=False, silence = False, return_std=False):


    init_params = {
        'f': np.mean(xdata),      # center frequency
        'A': abs(ydata[-1] - ydata[0]),  # amplitude scaling
        'kappa': 1e-3,
        'q': 1,                    # Fano asymmetry parameter
        'chi' : 1e-3, 
        'offset': np.mean(ydata),
        'slope': (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])              # slope of the background # TODO clean up all these bg nonsense
    }

    if verbose:
        print(f'initial guesses: {init_params}')

    return generic_fit(double_fano_bg_o1,
                       xdata, ydata,
                       init_params=init_params,
                       verbose=verbose,
                       silence=silence,
                       return_std=return_std)

def get_closest_res_val(flux, flux_list, res_list): 
    """Get the closest resonator value to the flux value"""
    flux_diff = np.abs(flux_list - flux)
    index = np.argmin(flux_diff)
    return res_list[index]

def round_if_number(x, ndigits=4):
    if isinstance(x, float):
        return round(x, ndigits)
    if isinstance(x, (list, tuple)):
        return [round(v, ndigits) if isinstance(v, float) else v for v in x]
    return x       

def rotate_iq(di, dq, angle):
    """
    Rotate the I/Q data by a given angle in radians.
    """
    di_rotated = di * np.cos(angle) - dq * np.sin(angle)
    dq_rotated = di * np.sin(angle) + dq * np.cos(angle)
    return di_rotated, dq_rotated

def dump_dict_h5(obj, h5group):
    """Write a (possibly nested) dict into an h5py Group."""
    for key, val in obj.items():
        if isinstance(val, dict):
            dump_dict_h5(val, h5group.require_group(key))      # recurse
        else:
            # convert scalars to 0-D NumPy arrays for portability
            h5group.create_dataset(key, data=val)

def avoided_crossing(x, x0, omega, A):
    """Avoided crossing model"""
    delta = x - x0
    return A / np.sqrt(omega **2 + delta **2)

def fit_avoided_crossing(xdata, ydata, weights = None, 
                         silence = False, verbose=False, return_std=False):
    """
    Fit the avoided crossing model to the data.
    """
    init_params = {
        'omega': dict(value = 1 / (np.mean(ydata) * 4), min = 0),
        'x0': dict(value = np.mean(xdata)),
        'A': dict(value = 1, min = 0)
    }

    return generic_fit(avoided_crossing, xdata, ydata, init_params, weights = weights, 
                       silence = silence, verbose=verbose, return_std=return_std)