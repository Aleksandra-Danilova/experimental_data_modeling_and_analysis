'''
    File contains examples of tasks that can be done with data:
    reading, modelling, processing, and analysis.
'''


from scipy.misc import derivative
from src.data import Data
from src.model import Model
from src.analysis import Analysis
from src.processing import Processing
import numpy as np
from random import uniform
from scipy.io import wavfile
from matplotlib import pyplot as plt
import cv2
from scipy.signal import convolve, convolve2d


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
    #plt.gcf().canvas.set_window_title("Impulse Response of the Cardiac Muscle")
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
    #plt.gcf().canvas.set_window_title("Rhythm Control Function")
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
        #plt.gcf().canvas.set_window_title("Convolution")
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


def read_image_from_dat(path, xlen, ylen):
    dat_file = np.fromfile(path, dtype='float32')
    dat_file_reshaped = np.reshape(dat_file, (xlen, ylen))
    plt.imshow(dat_file_reshaped, cmap='gray')
    plt.title(path)
    plt.show()
    return dat_file_reshaped   


# Inverse filter for .dat files
def inverse_filter_for_dat(dat, kernel, alpha=0):
        N = 1000
        M = 10
        new_image = list()
        H = Analysis(N, M).complex_spectrum(kernel, len(kernel), next_division=True)
        for i in range(221):  
            Y = Analysis(N, M).complex_spectrum(dat[i], 307, next_division=True)
            inverse_filter = Processing(N).inverse_filter(Y, H, N=307, M=200, alpha=alpha)
            new_image.append(inverse_filter)
        
        plt.imshow(new_image, cmap='gray')
        plt.title('New Image')
        plt.show()
        return new_image


# Edge detection by frequency filters
def get_edges(image, threshold, filter='lpf'):

    N = 1000
    m = 64
        
    # 1) binarize image
    binary_image = Processing(N).binarize_image(image, threshold)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary Image')
    plt.show()
    N = len(binary_image[0])
        
    # 2) get image erosion by applying Low Pass Filter (LPF)
    # to rows and columns
    if filter == 'lpf':
        # Frequences reduction with LPF 
        lpf = Processing(N).lpf(fc=0.1, dt=1, m=m, symmetrical_display=True, plot=True)
        lpf_rows = list()
        for row in binary_image:
            #filter = Analysis(N, M).filter_frequency(lpf, m=64, filter_type='LP', plot=True, plot_spectrum_independently=False, dt=1)
            conv_lpf = convolve(row, lpf, mode='same')#convolution(row, lpf, N=N, M=2*m+1, dt=1, plot=False)
            lpf_rows.append(conv_lpf)#[m:-1] + conv_lpf[0:m+1])#conv_lpf[2*m+1:-1])
        print(len(lpf_rows))
        print(len(lpf_rows[0]))
            
        plt.imshow(lpf_rows, cmap='gray')
        plt.title('Low Pass Filter Applied to Rows')
        plt.show()
            
        M = len(lpf_rows[0])
            
        erosion = list()
        lpf_np = np.array(lpf_rows)

        for i in range(N): 
            conv_lpf = convolve(lpf_np[:, i], lpf,mode='same')#convolution(lpf_np[:, i], lpf, N=M, M=2*m+1, dt=1, plot=False)
            erosion.append(conv_lpf)#[2*m+1:-1]#conv_lpf[m:-1] + conv_lpf[0:m+1])
                
        erosion_rotated = np.flip(np.rot90(erosion, k=-1), axis=1)
        plt.imshow(erosion_rotated, cmap='gray')
        plt.title('Erosion')
        plt.show()
            
        # 3) binarize erosion
        erosion_binary = Processing(N).binarize_image(erosion_rotated, threshold)
        plt.imshow(erosion_binary, cmap='gray')
        plt.title('Binary image')
        plt.show()
        
        
        # 4) subtract binarized erosion from original binarized image
        image_with_edges = list()
        for item1, item2 in zip(binary_image, erosion_binary):
            image_with_edges.append(item1 - item2)
            
        plt.imshow(image_with_edges, cmap='gray', vmin=0, vmax=255)
        plt.title('Edges')
        plt.show()
        
    elif filter == 'hpf':
        # Frequences reduction with HPF
        hpf = Processing(N).hpf(fc=0.1, dt=1, m=m, plot=True)
    
        hpf_rows = list()
        for row in binary_image:
            #filter = Analysis(N, N).filter_frequency(hpf, m=64, filter_type='HP', plot=True, plot_spectrum_independently=False, dt=1)
            conv_hpf = convolve(row, hpf, mode='same')#convolution(row, hpf, N=N, M=2*m+1, dt=1, plot=False)
            hpf_rows.append(conv_hpf)#[m:-1] + conv_hpf[0:m+1])#conv_hpf[2*m+1:-1])
            
        plt.imshow(hpf_rows, cmap='gray')
        plt.title('High Pass Filter Applied to Rows')
        plt.show()
            
        M = len(hpf_rows[0])
            
        erosion = list()
        hpf_np = np.array(hpf_rows)

        for i in range(N): 
            conv_hpf = convolve(hpf_np[:, i], hpf, mode='same')#convolution(hpf_np[:, i], hpf, N=M, M=2*m+1, dt=1, plot=False)
            erosion.append(conv_hpf)#[2*m+1:-1]#conv_hpf[m:-1] + conv_hpf[0:m+1])
        
        erosion_rotated = np.flip(np.rot90(erosion, k=-1), axis=1)
        plt.imshow(erosion_rotated, cmap='gray')
        plt.title('Erosion')
        plt.show()
            
        # 3) binarize erosion
        erosion_binary = Processing(N).binarize_image(erosion_rotated, 0.3)
        plt.imshow(erosion_binary, cmap='gray')
        plt.title('Edges')
        plt.show()





def main():
    # initialize variables
    
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
    #data_noise = model.noise(N, R, plot_graphs=False)
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
    
    #data_harm = model.harmonic(N, A0, f0, delta_t, plot_graphs=False)
    #dataX = model.harmonic(N, A2, f2, delta_t, plot_graphs=False)
    #data_polyharm = model.polyHarm(N, Ai, fi, delta_t, plot_graph=False)

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
    S
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
    '''
    
    
    # Shift image pixel values by constant
    # with following recalculation to grayscale
    '''
    try:
        path = 'src/grace.jpg'
        image = Data(N).read_file(path, ext='jpg')
         
        
        shift_2d = Model(N).shift_2D(image, C=30,
                                     new_filename='src/shifted_2d.jpg',
                                     write=True, show=True)
        
        multModel_2d = Model(N).multModel_2D(image, C=1.3,
                                             new_filename='src/multiplied_2d.jpg',
                                             write=True, show=True) 
        
    except:
        print('Error')
    
    recalculate_to_grayscale = Data(N).recalculate_to_grayscale(shift_2d,
                                                    new_filename='src/shifted_recalc.jpg',
                                                    write=True, show=True)
    
    recalculate_to_grayscale_mult = Data(N).recalculate_to_grayscale(multModel_2d,
                                                    new_filename='src/multiplied_recalc.jpg',
                                                    write=True, show=True)
    '''
    
    
    
    # Read .xcr file with ribs xray, where header of 2048 bytes is a text, ignore tail of 8192 bytes
    '''
    path = 'src/c12-85v.xcr'
    image = Data(N).read_file(path, ext='xcr', shape=(1024, 1024), start=2048, end=-8192, rot=True)
    #print(len(image))
    
    plt.imshow(image, cmap='gray')
    plt.title("xcr")
    plt.show()
    
    cv2.imwrite('src/xcr_recalc.jpg', image)
    image_jpg = cv2.imread('src/xcr_recalc.jpg', cv2.IMREAD_GRAYSCALE)
    #img = Data(N).write_to_xcr(image, 'src/xcr_recalc.bin')
    image_jpg.tofile('src/to_file.bin')
    
    
    #img = cv2.imread('src/to_file.bin', cv2.IMREAD_GRAYSCALE)
    #plt.imshow(img, cmap='gray')
    #plt.title("bin")
    #plt.show()
    '''
    
    
    
    #Resize images with scaling factors greater and less than 1
    #and with the nearest neighbour method / bilinear interpolation
    '''
    # Resize image grace 
    path = 'src/grace.jpg'
    image = Data(N).read_file(path, ext='jpg')
    
    resized_large = Data(N).resize_image(image[0], scale_factor=1.3)
    plt.imshow(resized_large, cmap='gray')
    plt.title("resized, scale_factor=1.3")
    plt.show()
    
    resized_small = Data(N).resize_image(image[0], scale_factor=0.7)
    plt.imshow(resized_small, cmap='gray')
    plt.title("resized, scale_factor=0.7")
    plt.show()
    
    resized_large_bilinear = Data(N).resize_image(image[0], scale_factor=1.3, method='bilinear')
    plt.imshow(resized_large_bilinear, cmap='gray')
    plt.title("resized, scale_factor=1.3")
    plt.show()
    
    resized_small_bilinear = Data(N).resize_image(image[0], scale_factor=0.7, method='bilinear')
    plt.imshow(resized_small_bilinear, cmap='gray')
    plt.title("resized, scale_factor=0.7")
    plt.show()
    
    
    # Resize ribs
    path = 'src/c12-85v.xcr'
    image = Data(N).read_file(path, ext='xcr', shape=(1024, 1024), start=2048, end=-8192, rot=True)

    plt.imshow(image, cmap='gray')
    plt.title("xcr")
    plt.show()
    
    resized_small = Data(N).resize_image(image, scale_factor=0.6)
    plt.imshow(resized_small, cmap='gray')
    plt.title("resized, scale_factor=0.6")
    plt.show()
    
    resized_small_bi = Data(N).resize_image(image, scale_factor=0.6, method='bilinear')
    plt.imshow(resized_small_bi, cmap='gray')
    plt.title("resized, scale_factor=0.6")
    plt.show()
    
    
    # Resize chest xray
    path = 'src/u0_2048x2500.xcr'
    image = Data(N).read_file(path, ext='xcr', shape=(2500, 2048), start=2048, end=-8192, rot=True)
    
    plt.imshow(image, cmap='gray')
    plt.title("xcr")
    plt.show()
    
    resized_small = Data(N).resize_image(image, scale_factor=0.6)
    
    plt.imshow(resized_small, cmap='gray')
    plt.title("resized, scale_factor=0.6")
    plt.show()
    
    resized_small_bil = Data(N).resize_image(image, scale_factor=0.6, method='bilinear')
    plt.imshow(resized_small_bil, cmap='gray')
    plt.title("resized, scale_factor=0.6")
    plt.show()
    '''

    
    
    # Get negative images
    '''
    # Negative grace image
    path = 'src/grace.jpg'
    image = Data(N).read_file(path, ext='jpg')
    negative_grace = Processing(N).get_negative(image[0])
    plt.imshow(negative_grace, cmap='gray')
    plt.title("Negative")
    plt.show()
    
    # Negative ribs xray
    path = 'src/c12-85v.xcr'
    image = Data(N).read_file(path, ext='xcr', shape=(1024, 1024), start=2048, end=-8192, rot=True)
    negative_xcr = Processing(N).get_negative(image)
    plt.imshow(negative_xcr, cmap='gray')
    plt.title("Negative")
    plt.show()
    
    # Negative chest xray
    path = 'src/u0_2048x2500.xcr'
    image = Data(N).read_file(path, ext='xcr', shape=(2500, 2048), start=2048, end=-8192, rot=True)
    negative_u0 = Processing(N).get_negative(image)
    plt.imshow(negative_u0, cmap='gray')
    plt.title("Negative")
    plt.show()
    '''
    
    # Get gamma correction
    '''
    path = 'src/img1.jpg'
    image = Data(N).read_file(path, ext='jpg')
    gamma1 = Processing(N).gamma_conversion(image[0], C=10, gamma=0.7)
    plt.imshow(gamma1, cmap='gray')
    plt.title("Gamma Correction, C=10, y=0.7")
    plt.show()
    
    path = 'src/img2.jpg'
    image = Data(N).read_file(path, ext='jpg')
    gamma2 = Processing(N).gamma_conversion(image[0], C=10, gamma=0.6)
    plt.imshow(gamma2, cmap='gray')
    plt.title("Gamma Correction, C=10, y=0.6")
    plt.show()
    
    path = 'src/img3.jpg'
    image = Data(N).read_file(path, ext='jpg')
    gamma3 = Processing(N).gamma_conversion(image[0], C=1, gamma=0.7)
    plt.imshow(gamma3, cmap='gray')
    plt.title("Gamma Correction, C=1, y=0.7")
    plt.show()
    
    path = 'src/img4.jpg'
    image = Data(N).read_file(path, ext='jpg')
    gamma4 = Processing(N).gamma_conversion(image[0], C=10, gamma=0.5)
    plt.imshow(gamma4, cmap='gray')
    plt.title("Gamma Correction, C=10, y=0.5")
    plt.show()
    '''
    
    # Get logarithmical transformation
    '''
    path = 'src/img1.jpg'
    image = Data(N).read_file(path, ext='jpg')
    log1 = Processing(N).logarithmic_transformation(image[0], C=0.1)
    plt.imshow(log1, cmap='gray')
    plt.title("Log transform, C=0.1")
    plt.show()
    
    path = 'src/img2.jpg'
    image = Data(N).read_file(path, ext='jpg')
    log2 = Processing(N).logarithmic_transformation(image[0], C=0.1)
    plt.imshow(log2, cmap='gray')
    plt.title("Log transform, C=0.1")
    plt.show()
    
    path = 'src/img3.jpg'
    image = Data(N).read_file(path, ext='jpg')
    log3 = Processing(N).logarithmic_transformation(image[0], C=1)
    plt.imshow(log3, cmap='gray')
    plt.title("Log transform, C=1")
    plt.show()
    
    path = 'src/img4.jpg'
    image = Data(N).read_file(path, ext='jpg')
    log4 = Processing(N).logarithmic_transformation(image[0], C=1)
    plt.imshow(log4, cmap='gray')
    plt.title("Log transform, C=1")
    plt.show()
    '''
    
    
    # Getting histogram equalization
    '''
    path = 'src/HollywoodLC.jpg'
    image = Data(N).read_file(path, ext='jpg')
    # Normalized histogram of source image
    p = Processing(N).normalized_histogram(image[0])
    cdf = Processing(N).CDF(image[0], p)
    eq = Processing(N).histogram_equalization(image[0], cdf)
    # Normalized histogram after histogram equalization
    p_eq = Processing(N).normalized_histogram(eq)
    cdf = Processing(N).CDF(eq, p_eq)
    
    path = 'src/img1.jpg'
    image = Data(N).read_file(path, ext='jpg')
    # Normalized histogram of source image
    p = Processing(N).normalized_histogram(image[0])
    cdf = Processing(N).CDF(image[0], p)
    eq = Processing(N).histogram_equalization(image[0], cdf)
    # Normalized histogram after histogram equalization
    p_eq = Processing(N).normalized_histogram(eq)
    cdf = Processing(N).CDF(eq, p_eq)
    
    
    path = 'src/img2.jpg'
    image = Data(N).read_file(path, ext='jpg')
    # Normalized histogram of source image
    p = Processing(N).normalized_histogram(image[0])
    cdf = Processing(N).CDF(image[0], p)
    eq = Processing(N).histogram_equalization(image[0], cdf)
    # Normalized histogram after histogram equalization
    p_eq = Processing(N).normalized_histogram(eq)
    cdf = Processing(N).CDF(eq, p_eq)

    
    path = 'src/img3.jpg'
    image = Data(N).read_file(path, ext='jpg')
    # Normalized histogram of source image
    p = Processing(N).normalized_histogram(image[0])
    cdf = Processing(N).CDF(image[0], p)
    eq = Processing(N).histogram_equalization(image[0], cdf)
    # Normalized histogram after histogram equalization
    p_eq = Processing(N).normalized_histogram(eq)
    cdf = Processing(N).CDF(eq, p_eq)
    
    
    path = 'src/img4.jpg'
    image = Data(N).read_file(path, ext='jpg')
    # Normalized histogram of source image
    p = Processing(N).normalized_histogram(image[0])
    cdf = Processing(N).CDF(image[0], p)
    eq = Processing(N).histogram_equalization(image[0], cdf)
    # Normalized histogram after histogram equalization
    p_eq = Processing(N).normalized_histogram(eq)
    cdf = Processing(N).CDF(eq, p_eq)
    '''
    
    
    # Compare images: orginal and resized one with interpolation
    '''
    path = 'src/grace.jpg'
    image = Data(N).read_file(path, ext='jpg')
    print(image[0].shape)
    
    resized_large = Data(N).resize_image(image[0], scale_factor=1.5)#, method='bilinear')
    print(resized_large.shape)
    plt.imshow(resized_large, cmap='gray')
    plt.title("resized, scale_factor=1.5")
    plt.show()
    
    resized_small = Data(N).resize_image(resized_large, scale_factor=1/1.5)#, method='bilinear')
    print(resized_small.shape)
    plt.imshow(resized_small, cmap='gray')
    plt.title("resized, scale_factor=1/1.5")
    plt.show()
    
    difference = Analysis(N, M).compare_images(image[0], resized_small)
    
    # Normalized histogram of source image
    p = Processing(N).normalized_histogram(difference)
    cdf = Processing(N).CDF(image[0], p)
    eq = Processing(N).histogram_equalization(difference, cdf)
    # Normalized histogram after histogram equalization
    p_eq = Processing(N).normalized_histogram(eq)
    cdf = Processing(N).CDF(eq, p_eq)
    
    negative_grace = Processing(N).get_negative(difference)
    plt.imshow(negative_grace, cmap='gray')
    plt.title("Negative")
    plt.show()
    '''

   
    
    # Detection and suppression of artifacts of anti-scattering grids in X-ray images
    # uncomment 'processing.detect_artefacts(image)' line to explore detection process
    '''
    # Artifacts for ribs xray
    path = 'src/c12-85v.xcr'
    image = Data(N).read_file(path, ext='xcr', shape=(1024, 1024), start=2048, end=-8192, rot=True)#[0:256, 0:256]
    plt.imshow(image, cmap='gray')
    plt.title("xcr")
    plt.show()
    #processing.detect_artefacts(image)
    processing.suppress_artifacts(image, fc1=0.25, fc2=0.35, m=32)
    

    # Artifacts for chest xray
    path = 'src/u0_2048x2500.xcr'
    image = Data(N).read_file(path, ext='xcr', shape=(2500, 2048), start=2048, end=-8192, rot=True)[0:256, 0:256]
    plt.imshow(image, cmap='gray')
    plt.title("xcr")
    plt.show()
    #processing.detect_artefacts(image)
    processing.suppress_artifacts(image, fc1=0.35, fc2=0.41, m=32)
    
    # Derivatives for chest xray image
    derivative = list()
    N = len(image[0])
    for i in range(0, N-1):
        derivative.append((image[0][i] - image[0][i + 1]))
    derivative.append(derivative[-1])
                
    acf = Analysis(N, M).acf(derivative, N)
    Xn_acf = Analysis(N, M).Fourier(acf, N)
    fourier = Analysis(N, M).spectrFourier(Xn_acf, N, acf, dt=1)
    '''
    
    
    
    
    
    # Modelling of noise: random, salt and pepper, both.
    # Then noise supression with median and arithmetic mean filters.
    '''
    path = 'src/MODELimage.jpg'
    image = Data(N).read_file(path, ext='jpg')
    print(np.amin(image[0]))
    
    M = 10
    R_imp = 255
    R_rnd = 100
    
    
    salt_and_pepper = Model(N).noise_image(image[0], noise_type='salt_pepper', M=M, R_imp=R_imp)
    random = Model(N).noise_image(image[0], noise_type='random', R_rnd=R_rnd)
    both = Model(N).noise_image(image[0], noise_type='both', M=M, R_imp=R_imp, R_rnd=R_rnd)
    
    #Processing(N).suppress_noise(np.array(salt_and_pepper), filter_size=3, filter_type='median')
    #Processing(N).suppress_noise(np.array(random), filter_size=3, filter_type='median')
    #Processing(N).suppress_noise(np.array(both), filter_size=3, filter_type='median')
    Processing(N).suppress_noise(np.array(salt_and_pepper), filter_size=3, filter_type='mean')
    Processing(N).suppress_noise(np.array(random), filter_size=3, filter_type='mean')
    Processing(N).suppress_noise(np.array(both), filter_size=3, filter_type='mean')
    '''
    
    
    # Inverse Fourier Transform
    '''
    # Example of harmonic function restoration with 1D inverse Fourier transform.
    data_harm = model.harmonic(N=1000, A0=100, f0=33, delta_t=0.001, plot_graphs=False)
    Xn_harm = Analysis(N, M).Fourier(data_harm, N)
    fourier_harm = Analysis(N, M).spectrFourier(Xn_harm, N, data_harm, type='harm', dt=0.001)
    Xn_harm_restored = Analysis(N, M).inverse_Fourier(data_harm, N)
    

    # Test image with rectangle inside for 2D inverse Fourier transform
    
    N = 256
    rectangle = np.zeros((N, N))#, dtype=int)
    n = N // 2
    for i in range(n - 10, n + 10 + 1):
        for j in range(n - 15, n + 15 + 1):
            rectangle[i, j] = 255
    plt.imshow(rectangle, cmap=plt.get_cmap('gray'))
    plt.title("Original")
    plt.show()    
    
    fourier_2d = Analysis(N, M).Fourier2D(rectangle)
    log1 = Processing(N).logarithmic_transformation(np.array(fourier_2d), C=10)
    plt.imshow(log1, cmap='gray')
    plt.title("Log transform, C=10")
    plt.show()
    
    inverse_fourier_2d = Analysis(N, M).inverse_Fourier2D(rectangle)
   
    
    # Test grace image for 2D inverse Fourier transform
    path = 'src/grace.jpg'
    image = Data(N).read_file(path, ext='jpg')[0]

    fourier_2d = Analysis(N, M).Fourier2D(image)
    
    log1 = Processing(N).logarithmic_transformation(np.array(fourier_2d), C=10)
    plt.imshow(log1, cmap='gray')
    plt.title("Log transform, C=10")
    plt.show()
    
    inverse_fourier_2d = Analysis(N, M).inverse_Fourier2D(image)
   
    # Inverse Fourier Transform for cardiogram
    ht_norm = impulse_response(N, dt=0.005)
    xt = rhythm_control(N, dt=0.005)
    y = convolution(xt, ht_norm, N=1000, M=200, dt=0.005)
    
    # complex spectrum of cardiogram (Re and Im)
    Y = Analysis(N, M).complex_spectrum(y, len(y), next_division=True)
    # complex spectrum of cardiac muscle function (Re and Im)
    H = Analysis(N, M).complex_spectrum(ht_norm, len(ht_norm), next_division=True)
    inverse_filter = Processing(N).inverse_filter(Y, H, N=1000, M=200)
    plt.plot(inverse_filter)
    plt.title('1D Inverse Fourier Transform')
    plt.show()
    
    alpha = 0.1
    noise = Model(N).noise(N, R=1, plot_graphs=True)
    additive_model = Model(N).addModel(y, noise, N, plot=True)
    Y = Analysis(N, M).complex_spectrum(additive_model, len(additive_model), next_division=True)
    inverse_filter = Processing(N).inverse_filter(Y, H, N=1000, M=200, alpha=alpha)
    plt.plot(inverse_filter)
    plt.title('1D Inverse Fourier Transform')
    plt.show()
    inverse_filter = Processing(N).inverse_filter(Y, H, N=1000, M=200, alpha=0.01)
    plt.plot(inverse_filter)
    plt.title('1D Inverse Fourier Transform')
    plt.show()
    '''
    


    # Restore blurry images with inverse filter
    '''
    # Reading data from .dat files
    kern76D = Data(N).read_file('src/kern76D.dat')
    # add extra zeros
    kern76D_xt = [kern76D[1][i] if i < len(kern76D[1]) else 0 for i in range(307)]

    blur307_221D = read_image_from_dat('src/blur307x221D.dat', 221, 307)    
    blur307_221D_N = read_image_from_dat('src/blur307x221D_N.dat', 221, 307)
    
    new_image = inverse_filter_for_dat(blur307_221D, kern76D_xt)
    new_image_N = inverse_filter_for_dat(blur307_221D_N, kern76D_xt, alpha=0.008)
    '''
    

  
           
        
    # Edge detection by frequency filters
    # Images without noise
    '''
    path_model = 'src/MODELimage.jpg'
    image = Data(N).read_file(path_model, ext='jpg')
    image = image[0]
    #image = Data(N).recalculate_to_grayscale(image, write=False, show=True)
    get_edges(image, 0.75)
    #get_edges(image, 0.75, 'hpf')
    '''
    '''
    path_grace = 'src/grace.jpg'
    image = Data(N).read_file(path_grace, ext='jpg')
    image = image[0]
    image = Data(N).recalculate_to_grayscale(image, write=False, show=True)
    get_edges(image, 0.6)
    #get_edges(image, 0.2, 'hpf')
    '''
    '''
    # Images with noise
    M = 2
    R_imp = 50
    R_rnd = 100
    
    both = np.array(Model(N).noise_image(image, noise_type='both', M=M, R_imp=R_imp, R_rnd=R_rnd))
    #both = Data(N).recalculate_to_grayscale(both, write=False, show=True)
    #get_edges(both, 0.5)
    #get_edges(both, 0.75, 'hpf')
    '''
    '''
    mean = Processing(N).suppress_noise(both, filter_size=11, filter_type='mean')
    #get_edges(mean, 0.5)
    get_edges(mean, 0.85, 'hpf')
    '''
    '''
    median = Processing(N).suppress_noise(both, filter_size=11, filter_type='median')
    #get_edges(median, 0.5)
    get_edges(median, 0.8, 'hpf')
    '''
    
    
    
   
    
if __name__ == "__main__":
   main()