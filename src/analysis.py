from src.data import Data
from src.model import Model
from matplotlib import pyplot as plt
from math import sqrt, pi, sin, cos
import warnings
warnings.filterwarnings("ignore")


class Analysis(Model, Data):
    def __init__(self, N, M):
        super(Model, self).__init__(N)
        self.M = M
          
           
    def __get_min(self, data):
        return min(data)
    
    
    def __get_max(self, data):
        return max(data)
    
    
    def _get_mean(self, data):
        return float(sum(data) / len(data))
    
    
    def _get_dispersion(self, data):
        means = self._get_mean(data)
        dispersion_values = sum([(num - means)**2 for num in data]) / len(data)
        return dispersion_values
        
        
    def _get_standard_deviation(self, data):
        dispersion_values = self._get_dispersion(data)
        standard_deviation_values = sqrt(dispersion_values)
        return standard_deviation_values
    
    
    def __get_asymmetry(self, data):
        means = self._get_mean(data)
        standard_deviation = self._get_standard_deviation(data)
        
        asymmetry_values = sum([(num - means)**3 for num in data]) / self.N
        asymmetry_coefficients = asymmetry_values / standard_deviation**3
            
        return asymmetry_values, asymmetry_coefficients
    
    
    def __get_excess(self, data):
        means = self._get_mean(data)
        standard_deviation = self._get_standard_deviation(data)
   
        excess_values = sum([(num - means)**4 for num in data]) / self.N
        kurtosis = excess_values / standard_deviation**4 - 3
            
        return excess_values, kurtosis
    
    
    def __get_mean_square(self, data):
        mean_square_values = sum([num**2 for num in data]) / self.N
        return mean_square_values
        
        
    def __get_rmse(self, data):
        mean_square_values = self.__get_mean_square(data)
        return sqrt(mean_square_values)
    
    
    def get_statistics(self, data):  
        stats = {"Min Value": self.__get_min(data),
                      "Max Value": self.__get_max(data),
                      "Mean Value": self._get_mean(data),
                      "Dispersion Value": self._get_dispersion(data),
                      "Standard Deviation Value": self._get_standard_deviation(data),
                      "Asymmetry Value and Coefficient": self.__get_asymmetry(data),
                      "Excess and Kurtosis": self.__get_excess(data),
                      "Mean Square Value": self.__get_mean_square(data),
                      "Root Mean Square Error": self.__get_rmse(data)}       
        return stats
    
    
    def __calculate_relative_errors(self, R, stats_results):
        relative_errors = list()
        for i in range(len(stats_results) - 1):
            for j in range(i + 1, len(stats_results)):
                if R == 1:
                    relative_errors.append((stats_results[i] - stats_results[j]) * 100)
                else:
                    relative_errors.append((stats_results[i] - stats_results[j]) / stats_results[j] * 100)      
        return relative_errors
    
    
    def __calculate_absolute_errors(self, stats_results):
        relative_errors = list()      
        for i in range(len(stats_results) - 1):
            for j in range(i + 1, len(stats_results)):
                relative_errors.append((stats_results[i] - stats_results[j]) * 100)
        return relative_errors
    
    
    def __compare_error_pairs(self, error_list, stationary=True):
        for i in range(len(error_list)):
            for j in range(i + 1, len(error_list)):
                if error_list[i] - error_list[j] >= 10:
                    stationary = False
                    break
        return stationary
    
    
    def __get_plot_stats(self, data, step, N):
        mean_values = list()
        means = list()
        std_values = list()
        std = list()
        trend_section = list()
        
        for section in range(0, N, step):
            trend_section = data[section:section+step]
            means.append(self.__get_mean(trend_section))
            std.append(self.__get_standard_deviation(trend_section))
        
        mean_values.append(means)
        std_values.append(std)
        means = list()
        std = list()
        
        return mean_values, std_values
                     
                     
    def estimate_stationarity(self, data, R):
        stationary_mean = True
        stationary_std = True
        stationary = list()
        step = self.N // self.M
        
        stats = self.__get_plot_stats(data, step, self.N)
        mean_values, std_values = stats[0], stats[1]
        
        # calculate errors
        mean_relative_error_list = list()
        std_relative_error_list = list()

        # unshifted data -> absolute errors
        if self.__get_mean(mean_values[0]) < 0.1:
            mean_relative_error_list = self.__calculate_absolute_errors(mean_values[0])
            std_relative_error_list = self.__calculate_absolute_errors(std_values[0])
        else:
            mean_relative_error_list = self.__calculate_relative_errors(R, mean_values[0])
            std_relative_error_list = self.__calculate_relative_errors(R, std_values[0])
        
        # compare pairs
        stationary_mean = self.__compare_error_pairs(mean_relative_error_list)
        stationary_std = self.__compare_error_pairs(std_relative_error_list)
        stationary = stationary_mean * stationary_std  
        
        if stationary == True:
            return "Stationary"
        else:
            return "Unstationary"
                

    # plot for probability density function
    def histogram(self, data, N, M, type=None):
        step = round(max(data) // M)
        data_temp = data
          
        if type=='trend':
            min_data = min(data)
            if min_data <= 0:
                shifted_data = [(i - min_data)  for i in data]
                data_temp = shifted_data
                step = round((max(data_temp)) // M)
            
        histogram = list()
        histogram_0 = list()
        y = sorted(data_temp)
        for i in range(0, round(max(y)), step):
            cnt = 0
            for x in range(0, N):
                if (data_temp[x] >= i and data_temp[x] < i + step):
                    cnt += 1
            histogram_0.append(cnt)
        
        step_x = N // M
        cnt_step_x = 0
        for i in range(0, N, step_x):
            for j in range(i, i + step_x):
                histogram.append(histogram_0[cnt_step_x])
            cnt_step_x += 1 
        
        plt.plot()
        plt.clf()
        plt.plot(histogram, label='histogram', alpha=0.9)
        plt.plot(data, label='data', alpha=0.4)
        plt.title("Histogram")
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.gcf().canvas.set_window_title("Histogram")
        plt.legend()
        plt.show()
        return histogram
    

    # graph of the normalized covariance (autocorrelation) function R(L)
    def acf(self, data, N):
        avg = self._get_mean(data)
        Rxx_list = list()
        for L in range(0, N):
            data_new = [(data[k] - avg) * (data[k+L] - avg) for k in range(0, N-L)]
            Rxx = sum(data_new) / N
            Rxx_list.append(Rxx)
        max_Rxx = max(Rxx_list)
        RL = [i / max_Rxx for i in Rxx_list]
    
        plt.plot()
        plt.clf()
        plt.plot(RL, label='Autocorrelation')
        plt.title("Autocorrelation")
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.gcf().canvas.set_window_title("Autocorrelation")
        plt.show()
        
        return RL
    
    
    # graph of cross-correlation (cross-covariance) function Rxy
    def ccf(self, dataX, dataY, N):
        avgX = self._get_mean(dataX)
        avgY = self._get_mean(dataY)
        Rxy_list = list()
        for L in range(0, N):
            data_new = [(dataX[k] - avgX) * (dataY[k+L] - avgY) for k in range(0, N-L)]
            Rxy = sum(data_new) / N
            Rxy_list.append(Rxy)

        plt.plot()
        plt.clf()
        plt.plot(Rxy_list, label='Cross-correlation')
        plt.title("Cross-correlation")
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.gcf().canvas.set_window_title("Cross-correlation")
        plt.show()
        
        return Rxy_list
    
    
    # calculate Fourier transform and its amplitude spectrum
    def Fourier(self, data, N, L=0):
        Re = list()
        Im = list()
        
        # set to 0 the last L values
        if L > 0:
            for i in range(N-L, N):
                data[i] = 0
                
        for n in range(N):
            Re_n = 0
            Im_n = 0
            for k in range(N):
                xk = data[k]
                Re_n += xk * cos(2*pi*n*k/N)
                Im_n += xk * sin(2*pi*n*k/N)
            Re.append(Re_n / N)
            Im.append(Im_n / N)
                
        # amplitude spectrum
        Xn_amplitude = [sqrt(Re[i]**2 + Im[i]**2) for i in range(N)]
        
        return Xn_amplitude
    
    
    # amplitude Fourier spectrum of N/2 length
    def spectrFourier(self, Xn, N, data, L=0, dt=None, type='orig', plot_graphs=True):
        N_half = N // 2
        F_N = N_half
        f_bound = F_N
        if dt is not None:
            f_bound = 1 / (2 * dt)
        rate = 2 * f_bound # sampling frequency
        delta_f = rate / N
        f = [n * delta_f for n in range(N_half)]
        Xn_amplitude = [Xn[i] for i in range(N_half)]
        
        if plot_graphs:
            if type == 'harm':
                type_name = 'Harmonic'
            elif type == 'polyharm':
                type_name = 'PolyHarmonic'
            elif type == 'add':
                type_name = 'Additive'
            elif type == 'trend':
                type_name = 'Trend'
            elif type == 'orig':
                type_name = 'Original'
                
            fig, axis = plt.subplots(1, 2, figsize=(10, 5), num=type_name+" Function & Fourier Amplitude Spectrum")
            fig.suptitle(type_name + f" Function & Fourier Amplitude Spectrum (L={L})")
            if dt == None:
                axis[0].plot([i for i in range(N)], data)
            else:
                axis[0].plot([i * dt for i in range(N)], data)
            axis[1].plot(f, Xn_amplitude)
            
            axis[0].set(xlabel='t', ylabel='x(t)')
            axis[1].set(xlabel='f [Hz]', ylabel='|Xn|')
            fig.tight_layout()  
            plt.show()  
                   
        return f, Xn_amplitude
    
    
    # Calculation of the frequency characteristic of filters by elementwisely
    # multiplication of the calculated spectrum values ​​by the transform length (2*m+1)
    def filter_frequency(self, data, m, filter_type='LP', plot=True, plot_spectrum_independently=True, dt=0.002):
        amplitude = self.Fourier(data, len(data))
        spectrum = list()
        if plot_spectrum_independently:
            spectrum = self.spectrFourier(amplitude, len(data), data, L=0, dt=0.002, plot_graphs=True)
        else:
            spectrum = self.spectrFourier(amplitude, len(data), data, L=0, dt=0.002, plot_graphs=False)
        loper = 2 * m + 1
        freq = [el * loper for el in spectrum[1]]
        rate = 1 / (2 * dt)
        delta_f = rate / m
        axis_x = [n * delta_f for n in range(m)]
    
        if plot:
            plt.plot(axis_x, freq)
            plt.title(f"Transfer Function {filter_type} filter")
            plt.xlabel("Frequency [Hz]")
            plt.gcf().canvas.set_window_title(f"Transfer Function {filter_type} filter")
            plt.show()
        
        return freq