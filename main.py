from src.data import Data
from src.model import Model
from src.analysis import Analysis
from src.processing import Processing
import numpy as np
from random import uniform
from scipy.io import wavfile
from matplotlib import pyplot as plt


# Additive Models
def get_additive_models(N=1000):
    model = Model(N)
    
    # linear trend and harmonic process
    data1 = model.trend(a=0.3, b=20, alpha=1, plot_graphs=False)[0]
    data2 = model.harmonic(N, A0=5, f0=50, delta_t=0.001, update=False, plot_graphs=False)
    
    # exponential trend and random noise
    data3 = model.trend(a=1, b=10, alpha=0.003, plot_graphs=False)[2]
    data4 = model.noise(N, R=10, plot_graphs=False)
    
    additive_model_12 = model.addModel(data1, data2, N, plot=True)
    additive_model_34 = model.addModel(data3, data4, N, plot=True)
    return additive_model_12, additive_model_34
    
    
# Remove (non-)liner trends from additive models
def remove_trends(additive_model_1, additive_model_2, N=1000, W=10):
    processing = Processing(N)
    
    # Remove linear trend in additive model
    processing.anti_trend_linear(additive_model_1, N)

    # Remove non-linear trend in additive model
    processing.anti_trend_nonlinear(additive_model_2, N, W)
    processing.anti_trend_nonlinear(additive_model_2, N, W=50, plot=True)
    
        
# Analysis of Fourier spectrum for harmonic and polyharmonic processes
def fourier_spectr_harm_polyharm(Ai, fi, A0=100, f0=33, N=1000, M=10, delta_t=0.001):
    model = Model(N)
    analysis = Analysis(N, M)
      
    data_harm = model.harmonic(N, A0, f0, delta_t, plot_graphs=False)
    data_polyharm = model.polyHarm(N, Ai, fi, delta_t, plot_graph=False)
    
    Xn_harm = analysis.Fourier(data_harm, N)
    Xn_polyharm = analysis.Fourier(data_polyharm, N)
    fourier_harm = analysis.spectrFourier(Xn_harm, N, data_harm, type='harm', dt=0.001)
    fourier_polyharm = analysis.spectrFourier(Xn_polyharm, N, data_polyharm, type='polyharm', dt=0.001)
    

# Examples of Fourier Amplitutude Spectrum
# for functions like additive model, linear trend and impulse
def get_fourier_spectr_examples(N=1000, M=10, a=1, b=10, alpha=0.0095, 
                                R=100, A0=100, f0=33, delta_t=0.001): 
    model = Model(N)
    analysis = Analysis(N, M)  
    
    # additive model: noise + harmonic
    noise = model.noise(N, R, plot_graphs=False)
    data_harm = model.harmonic(N, A0, f0, delta_t, plot_graphs=False)
    additive_model = model.addModel(noise, data_harm, N, plot=False)
    Xn_add = analysis.Fourier(additive_model, N)
    fourier_add = analysis.spectrFourier(Xn_add, N, additive_model, type='add')
    
    # linear trend
    trend = model.trend(a, b, alpha, plot_graphs=False)[0]
    Xn_trend = analysis.Fourier(trend, N)
    fourier_trend = analysis.spectrFourier(Xn_trend, N, trend, type='trend')
    
    # shifted data: horizontal line
    data = [0 for i in range(N)]
    offset = 100
    constant_shift = model.shift(data, C=offset, plot_graphs=False)
    Xn_trend = analysis.Fourier(constant_shift, N)
    fourier_trend = analysis.spectrFourier(Xn_trend, N, constant_shift, type='orig')
    
    # impulse
    data_noise = model.noise(N, R, plot_graphs=False)
    impulse_noise = model.impulse_noise(data_noise, N, M=1, R=1, plot=False, type='template')
    shifted_data = model.shift(impulse_noise, C=1, plot_graphs=False)
    Xn_imp = analysis.Fourier(shifted_data, N)
    fourier_imp = analysis.spectrFourier(Xn_imp, N, shifted_data)


def get_fourier_amplitude_spectrum_harm_polyharm(Ai, fi, A0=100, f0=33, 
                                                 L_list=[24, 124, 224],
                                                 N=1024, M=10, delta_t=0.001):
    # Fourier amplitude spectrum for harmonic and polyharmonic processes
    # of length N multiplied by rectangular window of length (N-L)
    analysis = Analysis(N, M)
    data_harm = analysis.harmonic(N, A0, f0, delta_t, plot_graphs=False)
    data_polyharm = analysis.polyHarm(N, Ai, fi, delta_t, plot_graph=False)
    
    for L in L_list:
        Xn_harm = analysis.Fourier(data_harm, N, L)
        Xn_polyharm = analysis.Fourier(data_polyharm, N, L)
        fourier_harm = analysis.spectrFourier(Xn_harm, N, data_harm, L, type='harm')
        fourier_polyharm = analysis.spectrFourier(Xn_polyharm, N, data_polyharm, L, type='polyharm')
        
        
def get_dependence_std_from_amount(N=1000, R=30, M=120):
    processing = Processing(N)
    t = [i for i in range(1, M, 10)]
    xt = list()
    for m in t:
        data = generate_processes(N, R, m)
        xt.append(processing.antiNoise(data, N, m, plot_graph=False)[1])
        
    plt.plot(t, xt)
    plt.title("Dependence of \u03C3 from M")
    plt.xlabel("M")
    plt.ylabel("\u03C3")
    plt.gcf().canvas.set_window_title("Dependence of \u03C3 from M")
    plt.show()
        
    return xt
    
    
# Anti-Noise Example
def generate_processes(N=1000, R=30, M=[1, 10, 100, 10000], type='noise'):
    data = list()
    model = Model(N)
    if M == 1:
        if type == 'noise':
            data = Model(N).noise(N, R, plot_graphs=False)
        elif type == 'harm':
            data = model.harmonic(N, A0=10, f0=5, delta_t=0.001, update=False, plot_graphs=False)               
    else:
        if type == 'noise':
            for i in range(M):
                data.append(Model(N).noise(N, R, plot_graphs=False))
        elif type == 'harm':
            for i in range(M):
                data.append(Model(N).harmonic(N, A0=10, f0=5, delta_t=0.001, update=False, plot_graphs=False))
    return data
    
    
def anti_noise(N=1000, M=[1, 10, 100, 10000], R=30):
    model = Model(N)
    processing = Processing(N)
    # Anti-Noise example for additive model of noise and harmonical process
    for amount in M:
        noise = generate_processes(N, R, amount, type='noise')
        harm = generate_processes(N, R, amount, type='harm')
        additive_model = list()
        if amount == 1:
            additive_model = model.addModel(noise, harm, N, plot=False)
            processing.antiNoise(additive_model, N, amount)
        elif amount > 1:
            for j in range(amount):
                additive_model.append(model.addModel(noise[j], harm[j], N, plot=False))
            processing.antiNoise(additive_model, N, amount)
    
    
def analyze_dat(N=1000, M=10, filename='pgp_2ms.dat', plot=True):
    analysis = Analysis(N, M)
    # Reading data from .dat file and analysis of its spectrum / amplitudes / frequences
    dat_file = Data(N).read_file(filename)
    dat_file_t = dat_file[0] # abscissa
    dat_file_xt = dat_file[1]
    Xn_dat = analysis.Fourier(dat_file_xt, len(dat_file_xt))
    fourier_dat = analysis.spectrFourier(Xn_dat, len(dat_file_xt), dat_file_xt, L=0, dt=0.002, plot_graphs=plot)
    return dat_file_xt, fourier_dat
    
    
# Additive Model: elementwise multiplication
def elementwise_multiplication(N=1000): 
    model = Model(N)
    data1 = model.trend(a=0.3, b=20, alpha=-20*0.005, plot_graphs=False)[2]
    data2 = model.harmonic(N, A0=1, f0=7, delta_t=0.005, update=False, plot_graphs=False)
    additive_model_12 = model.multModel(data1, data2, N, plot_graphs=True)
  
  
'''
    First approximation of the cardiogram model with a duration of 4 seconds
    using the convolution of the impulse response of the heart muscle model
    and the rhythm control function
'''
def impulse_response(N=1000, dt=0.005):
    # Impulse response of a linear model of the cardiac muscle
    # is a function of a multiplicative model h(t) = multModel(h1, h2, M,...),
    # where h1(t) is a harmonical process and h2(t) is a decreasing exponential trend 
    model = Model(N)
    h1t = model.harmonic(N, A0=1, f0=7, delta_t=dt, update=False, plot_graphs=False)
    h2t = model.trend(a=1, b=1, alpha=-30*dt, plot_graphs=False)[2]
    ht = model.multModel(h1t, h2t, N, plot_graphs=False)
    max_ht = max(ht)
    ht_norm = [h / max_ht * 120 for h in ht]
    plt.plot([i * dt for i in range(N)], ht_norm)
    plt.title("Impulse Response of the Cardiac Muscle")
    plt.xlabel("t")
    plt.ylabel("h(t)")
    plt.gcf().canvas.set_window_title("Impulse Response of the Cardiac Muscle")
    plt.show()
    return ht_norm
    
    
def rhythm_control(N=1000, dt=0.005):
    # Rhythm control function x(t) is set in the form of four pulses
    # of minimum duration with amplitudes 1+-0.1 following at regular intervals
    Rs = 0.1
    R = 1
    M = 200
    xt = [uniform(R-Rs, R+Rs) if t % M == 0 and t != 0 else 0 for t in range(N)]
    plt.clf()
    plt.plot([i * dt for i in range(N)], xt)
    plt.title("Rhythm Control Function")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.gcf().canvas.set_window_title("Rhythm Control Function")
    plt.show()  
    return xt
    
    
def convolution(xt, ht_norm, N=1000, M=200, dt=0.005, plot=True):
    # Discrete convolution of functions x(t) and h(t),
    # where last M values are dropped
    yk = list()
    for k in range(N):
        temp = 0
        for m in range(M):
            temp += xt[k-m]*ht_norm[m]
        yk.append(temp)

    if plot:
        plt.clf()
        plt.plot([i * dt for i in range(N)], yk)
        plt.title("Convolution")
        plt.xlabel("t")
        plt.ylabel("yk")
        plt.gcf().canvas.set_window_title("Convolution")
        plt.show()
    return yk
   

def filters(N=1000, M=10, fc=50, dt=0.002, m=64, fc1=35, fc2=75):
    analysis = Analysis(N, M)
    processing = Processing(N)
    # Filters:
    # Low Pass Filter (LPF)   
    lpf = processing.lpf(fc, dt, m, symmetrical_display=True, plot=True)
    analysis.filter_frequency(lpf, m, filter_type='LP')
    
    # High Pass Filter (HPF)
    hpf = processing.hpf(fc, dt, m, plot=True)
    analysis.filter_frequency(hpf, m, filter_type='HP')
    
    # Band Pass Filter (BPF)
    bpf = processing.bpf(fc1, fc2, dt, m, plot=True)
    analysis.filter_frequency(bpf, m, filter_type='BP')
    
    # Band Stop Filter (BSF)
    bsf = processing.bsf(fc1, fc2, dt, m, plot=True)
    analysis.filter_frequency(bsf, m, filter_type='BS')
        

def draw_plots_for_freq_filter_reduction(spectrum, filter, conv, fourier_dat, N=1000, dt=0.002):
    fig, axis = plt.subplots(3, 2, figsize=(15, 8))
    axis[0][0].plot([i * dt for i in range(N)], spectrum[0])
    axis[0][1].plot(spectrum[1][0], spectrum[1][1])
    axis[0][0].set(xlabel='t', ylabel='x(t)')
    axis[0][1].set(xlabel='f [Hz]', ylabel='|Xn|')
    axis[0][0].set_title("Data")
    axis[0][1].set_title("Fourier Amplitude Spectrum")

    rate = 1 / (2 * dt)
    delta_f = rate / len(filter)
    axis_x = [n * delta_f for n in range(len(filter))]
    
    axis[1][0].plot(axis_x, filter)
    axis[1][1].plot([i * dt for i in range(N)], conv)
    axis[1][0].set(xlabel='Frequency [Hz]')
    axis[1][1].set(xlabel='t', ylabel='yk')
    axis[1][0].set_title("Transfer Function Filter")
    axis[1][1].set_title("Convolution")
    
    axis[2][0].plot(fourier_dat[0], fourier_dat[1])
    axis[2][0].set(xlabel='f [Hz]', ylabel='|Xn|')
    axis[2][0].set_title("Fourier Amplitude Spectrum")
    fig.tight_layout()  
    plt.show()
    
       
def pw(signal_1d, coef1, n1, n2, coef2, n3, n4, N):
    data2 = list()
    oscillogram = list()
    
    for i in range(N):
        if i >= n1 and i <= n2:
            data2.append(signal_1d[i] * coef2)
            oscillogram.append(max(signal_1d[n1:n2+1]))
        elif i >= n3 and i <= n4:            
            data2.append(signal_1d[i] * coef1)
            oscillogram.append(max(signal_1d[n3:n4+1]))
        else:
            data2.append(0)
            oscillogram.append(0)
    return data2, oscillogram

    


def main():
    N = 1000
    #data = Data(N).t
    
    a = 1
    alpha = 0.0095
    b = 10
    model = Model(N)
    
    R = 1000  # R >= 1
    R_impulse = 500
    
    M = 10
    #dataX = model.customer_noise(N, R, plot_graphs=False)
    #dataX = model.noise(N, R, plot_graphs=False)
    #data = model.trend(a, b, alpha, plot_graphs=False)[0]
    data_noise = model.noise(N, R, plot_graphs=False)
    #dataX = model.noise(N, R, plot_graphs=False)
    #data = model.customer_noise(N, R, plot_graphs=False)
    
    analysis = Analysis(N, M)
    #analysis_stats = analysis.get_statistics(data)
    #for key, value in analysis_stats.items():
    #    print(f"{key}: {value}\n")
    
    #stationary = analysis.estimate_stationarity(data, R)
    #print(stationary)
    
    #shifted_data = model.shift(data, C=1000, plot_graphs=False)
    #print('Stationarity estimation for shifted data:')
    #stationary = Analysis(N, M).estimate_stationarity(shifted_data, R)
    #print(stationary)
    
    #model.impulse_noise(data, N, M, R, type='template', plot_graphs=True)
    
    A0 = 100
    f0 = 33
    A1 = 15
    f1 = 5
    A2 = 10#25
    f2 = 170
    delta_t = 0.001
    Ai = [A0, A1, A2]
    fi = [f0, f1, f2]
    
    data_harm = model.harmonic(N, A0, f0, delta_t, plot_graphs=False)
    #dataX = model.harmonic(N, A2, f2, delta_t, plot_graphs=False)
    data_polyharm = model.polyHarm(N, Ai, fi, delta_t, plot_graph=False)

    #analysis.histogram(data, N, M)
    
    #analysis.acf(data, N)
    #analysis.ccf(dataX, data, N)
    
    processing = Processing(N)
    '''
    #processing.antiShift(shifted_data)
    data_noise = model.noise(N, R, plot_graphs=False)
    impulse = model.impulse_noise(data_harm, N, M, R_impulse, type='data', plot=False)
    impulse_noise = model.impulse_noise(data_noise, N, M, R_impulse, type='data', plot=False)
    processing.anti_spike(impulse_noise, N, R_impulse)
    processing.anti_spike(impulse, N, R_impulse)
    '''
    
    '''
    # Additive models
    additive_models = get_additive_models()
    additive_model_12 = additive_models[0] # linear trend and harmonic process
    additive_model_34 = additive_models[1] # exponential trend and random noise
    
    # Remove (non-)linear trends in additive models
    remove_trends(additive_model_12, additive_model_34, N=1000)
    '''
    
    
    # Analysis of Fourier spectrum for harmonic and polyharmonic processes
    fourier_spectr_harm_polyharm(Ai, fi, A0=100, f0=33, N=1000, M=10, delta_t=0.001)
    
    get_fourier_spectr_examples(N=1000, M=10, a=1, b=10, alpha=0.0095, 
                                R=100, A0=100, f0=33, delta_t=0.001)
      
    # Fourier amplitude spectrum for harmonic and polyharmonic processes
    # of length N multiplied by rectangular window of length (N-L)
    get_fourier_amplitude_spectrum_harm_polyharm(Ai, fi, A0=100, f0=33, 
                                                 L_list=[24, 124, 224],
                                                 N=1024, M=10, delta_t = 0.001)
    
    R = 30
    N = 1000
    M = [1, 10, 100]
    for amount in M:
        data = generate_processes(N, R, amount, type='noise')
        processing.antiNoise(data, N, amount)
        
    get_dependence_std_from_amount(N=1000, R=30, M=120)
    
    anti_noise(N=1000, M=[1, 10, 100], R=30)
    
    analyze_dat(N=1000, M=10, filename='pgp_2ms.dat')
    
    elementwise_multiplication(N=1000) # Additive Model: elementwise multiplication
    
    ht_norm = impulse_response(N, dt=0.005)
    xt = rhythm_control(N, dt=0.005)
    convolution(xt, ht_norm, N=1000, M=200, dt=0.005)
    
    
    fc = 50
    dt = 0.002
    m = 64
    fc1 = 35
    fc2 = 75
    
    filters(N, M, fc=50, dt=0.002, m=64, fc1=35, fc2=75)
    
    # read .dat file to reduce frequences in spectrum with filters
    spectrum = analyze_dat(N=1000, M=10, filename='pgp_2ms.dat', plot=False)
    
    # Frequences reduction with LPF 
    lpf = processing.lpf(fc=10, dt=0.002, m=64, symmetrical_display=True, plot=False)
    filter = analysis.filter_frequency(lpf, m=64, filter_type='LP', plot=False, plot_spectrum_independently=False, dt=0.002)
    conv_lpf = convolution(spectrum[0], lpf, N=1000, M=2*m+1, dt=0.002, plot=False)
    Xn_dat = analysis.Fourier(conv_lpf, len(conv_lpf))
    fourier_dat = analysis.spectrFourier(Xn_dat, len(conv_lpf), conv_lpf, L=0, dt=0.002, plot_graphs=False)
    draw_plots_for_freq_filter_reduction(spectrum, filter, conv_lpf, fourier_dat, N=1000, dt=0.002)
    
    # Frequences reduction with HPF 
    hpf = processing.hpf(fc=40, dt=0.002, m=64, plot=False)
    filter = analysis.filter_frequency(hpf, m=64, filter_type='HP', plot=False, plot_spectrum_independently=False)
    conv_hpf = convolution(spectrum[0], hpf, N=1000, M=2*m+1, dt=0.002, plot=False)
    Xn_dat = analysis.Fourier(conv_hpf, len(conv_hpf))
    fourier_dat = analysis.spectrFourier(Xn_dat, len(conv_hpf), conv_hpf, L=0, dt=0.002, plot_graphs=False)
    draw_plots_for_freq_filter_reduction(spectrum, filter, conv_hpf, fourier_dat, N=1000, dt=0.002)
    
    # Frequences reduction with BPF   
    bpf = processing.bpf(fc1=35, fc2=75, dt=0.002, m=64, plot=False)
    filter = analysis.filter_frequency(bpf, m=64, filter_type='BP', plot=False, plot_spectrum_independently=False)
    conv_bpf = convolution(spectrum[0], bpf, N=1000, M=2*m+1, dt=0.002, plot=False)
    Xn_dat = analysis.Fourier(conv_bpf, len(conv_bpf))
    fourier_dat = analysis.spectrFourier(Xn_dat, len(conv_bpf), conv_bpf, L=0, dt=0.002, plot_graphs=False)
    draw_plots_for_freq_filter_reduction(spectrum, filter, conv_bpf, fourier_dat, N=1000, dt=0.002)    
    
    # Frequences reduction with BSF 
    bsf = processing.bsf(fc1=10, fc2=75, dt=0.002, m=64, plot=False)
    filter = analysis.filter_frequency(bsf, m=64, filter_type='BS', plot=False, plot_spectrum_independently=False)
    conv_bsf = convolution(spectrum[0], bsf, N=1000, M=2*m+1, dt=0.002, plot=False)
    Xn_dat = analysis.Fourier(conv_bsf, len(conv_bsf))
    fourier_dat = analysis.spectrFourier(Xn_dat, len(conv_bsf), conv_bsf, L=0, dt=0.002, plot_graphs=False)
    draw_plots_for_freq_filter_reduction(spectrum, filter, conv_bsf, fourier_dat, N=1000, dt=0.002)
    
    
   
if __name__ == "__main__":
   main()