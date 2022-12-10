from src.analysis import Analysis
from math import pi, sin, cos
from matplotlib import pyplot as plt


class Processing(Analysis):
    def __init__(self, N):
        super(Analysis, self).__init__(N)
        
        
    def antiShift(self, data):
        avg = self._get_mean(data)
        unshifted_data = [xt - avg for xt in data]
        
        fig, axis = plt.subplots(1, 2, figsize=(5, 3), num="Shifted & Anti-Shift Functions")
        fig.suptitle("Shifted & Anti-Shift Functions")
        axis[0].plot(self.t, data)
        axis[1].plot(self.t, unshifted_data)
        for ax in axis.flat:
            ax.set(xlabel='t', ylabel='x(t)')
        fig.tight_layout()  
        plt.show()  
        
        return unshifted_data
        
        
    # Detection and removal of implausible values
    # ​​outside the specified range R in additive data models
    def anti_spike(self, data, N, R, rs=0.1):
        Rs = R*rs
        unspiked_data = list()
        
        if abs(data[0]) > R - Rs:
            unspiked_data.append((data[1] + data[2]) * 0.5)
        else:
            unspiked_data.append(data[0])
            
        for k in range(1, N-1):
            if abs(data[k]) > R - Rs:
                xk_hat = (data[k-1] + data[k+1]) * 0.5
                unspiked_data.append(xk_hat)   
            else:
                unspiked_data.append(data[k]) 
        
        if abs(data[N-1]) > R - Rs:
            unspiked_data.append((data[N-2] + data[N-3]) * 0.5)
        else:
            unspiked_data.append(data[N-1])

        fig, axis = plt.subplots(1, 2, num="Impulse & Anti-Spiked Functions")
        fig.suptitle("Impulse & Anti-Spiked Functions")
        axis[0].plot(self.t, data)
        axis[1].plot(self.t, unspiked_data)
        for ax in axis.flat:
            ax.set(xlabel='t', ylabel='x(t)')
        fig.tight_layout()  
        plt.show()  
        
        return unspiked_data
    
    
    # Removing linear trend in a model by
    # calculation of the first derivative of additive model's data
    def anti_trend_linear(self, data, N, plot=True):
        antiTrend = list()
        for i in range(0, N-1):
            antiTrend.append((data[i] - data[i + 1]))
        antiTrend.append(antiTrend[-1])    
        
        if plot:
            fig, axis = plt.subplots(1, 2, figsize=(8, 4), num="Linear Trend & Anti-Trend Functions")
            fig.suptitle("Linear Trend & Anti-Trend Functions")
            axis[0].plot(self.t, data)
            axis[1].plot(self.t, antiTrend)
            for ax in axis.flat:
                ax.set(xlabel='t', ylabel='x(t)')
            fig.tight_layout()  
            plt.show()  
            
        return antiTrend
    
    
    # Average values ​​in the window, 
    # referred to the beginning of the window
    def __sliding_average(self, data, W):
        xnW = 0
        for k in range(0, W):
            xnW += data[k]
        xnW = xnW / W
        return xnW
        
    
    # Removing non-linear trend in a model by
    # selection of the trend component by the sliding average method and subsequent
    # element-by-element subtraction of the selected trend from the data of the additive model
    def anti_trend_nonlinear(self, data, N, W, plot=True):
        if W < 10: # sliding window size
            W = 10
            print('Parameter M must at least 10. Updated W is {W}.')
         
        antiTrend = list()
        for i in range(0, N-W, W):
            xnW = self.__sliding_average(data[i:i+W], W)
            for j in range(i, i+W):   
                antiTrend.append(data[j] - xnW)
                
        for i in range(len(antiTrend), N, 1):
            antiTrend.append(data[i])
            
        if plot:
            fig, axis = plt.subplots(1, 2, figsize=(8, 4), num="NonLinear Trend & Anti-Trend Functions")
            fig.suptitle("NonLinear Trend & Anti-Trend Functions")
            axis[0].plot(self.t, data)
            axis[1].plot(self.t, antiTrend)
            for ax in axis.flat:
                ax.set(xlabel='t', ylabel='x(t)')
            fig.tight_layout()  
            plt.show()  
            
        return antiTrend
    
    
    def __add_elementwisely(self, list1, list2):
        return list(map(sum, zip(list1 + [0,]*(len(list2)-len(list1)),
                            list2 + [0,]*(len(list1)-len(list2)))))
    
    
    # Random noise suppression function by accumulation method by elementwisely addition
    # and averaging of M random noise implementations x(t)_m of length N
    def antiNoise(self, data, N, M, plot_graph=True):
        xt = list()
        
        if M == 1:
            xt = data
        else:
            xt_temp = self.__add_elementwisely(data[0], data[1])
            for i in range(2, M):
                xt_temp = self.__add_elementwisely(xt_temp, data[i])
            xt = [element / M for element in xt_temp]
        
        std = self._get_standard_deviation(xt)
        
        if plot_graph:
            plt.plot(xt)
            plt.title(f"Anti-Noise (M={M}, \u03C3={round(std, 1)})")
            plt.xlabel("t")
            plt.ylabel("x(t)")
            plt.gcf().canvas.set_window_title(f"Anti-Noise (M={M}, \u03C3={round(std, 1)})")
            plt.show()
            
        return xt, std
    
    
    # Calculation of the impulse response or weight function
    # of the Low Pass Filter (LPF) with smoothing by the Potter window
    def lpf(self, fc, dt=1, m=64, symmetrical_display=True, plot=True):
        lpw = list()
        d = [0.35577019, 0.2436983, 0.07211497, 0.00630165]
        # rectangular part weights
        fact = float(2 * fc) * dt
        lpw.append(fact)
        arg = fact * pi
        for i in range(1, m + 1):
            lpw.append(sin(arg * i) / (pi * i))
        # trapeziod smoothing at the end
        lpw[m] /= 2.0
        # P310 smoothing window
        sumg = lpw[0]
        for i in range(1, m + 1):
            sum = d[0]
            arg = pi * i / m
            for k in range(1, 4):
                sum += 2.0 * d[k] * cos(arg*k)
            lpw[i] *= sum
            sumg += 2 * lpw[i]
        for i in range(0, m + 1):
            lpw[i] /= sumg
        
        time = [i for i in range(0, m + 1)]
        lpw_result = list()
        if symmetrical_display:
            time = [i for i in range(0, 2 * m + 1)]
            lpw_result = lpw[:0:-1] + lpw
        else:
            lpw_result = lpw

        if plot:
            plt.plot(time, lpw_result)
            plt.title("Potter Low Pass Filter Inverse Fourier Transform (LPF weights)")
            plt.xlabel("t")
            plt.gcf().canvas.set_window_title("Potter Low Pass Filter Inverse Fourier Transform (LPF weights)")
            plt.show()
        return lpw_result
    
   
    # Weights calculation using LPF
    # High Pass Filter (HPF)
    def hpf(self, fc, dt, m, plot=True):
        # hpw - weights for HPF
        hpw = list()
        lpw = self.lpf(fc, dt, m, plot=False)
        loper = 2 * m + 1
        for k in range(0, loper):
            if k == m:
                hpw.append(1.0 - lpw[k])
            else:
                hpw.append(-lpw[k])

        if plot:
            plt.plot(hpw)
            plt.title("High Pass Filter weights")
            plt.xlabel("t")
            plt.gcf().canvas.set_window_title("High Pass Filter weights")
            plt.show()
            
        return hpw
    
    
    # Band Pass Filter (BPF)
    def bpf(self, fc1, fc2, dt, m, plot=True):
        # bpw - weights for BPF;
        # fc1 < fc2
        bpw = list()
        lpw1 = self.lpf(fc1, dt, m, plot=False)
        lpw2 = self.lpf(fc2, dt, m, plot=False)
        loper = 2 * m + 1
        for k in range(0, loper):
            bpw.append(lpw2[k] - lpw1[k])
            
        if plot:
            plt.plot(bpw)
            plt.title("Band Pass Filter weights")
            plt.xlabel("t")
            plt.gcf().canvas.set_window_title("Band Pass Filter weights")
            plt.show()
            
        return bpw
    
    
    # Band Stop Filter (BSF)
    def bsf(self, fc1, fc2, dt, m, plot=True):
        # bsw - weights for BSF;
        # fc1 < fc2
        bsw = list()
        lpw1 = self.lpf(fc1, dt, m, plot=False)
        lpw2 = self.lpf(fc2, dt, m, plot=False)
        loper = 2 * m + 1
        
        for k in range(0, loper):
            if k == m:    
                bsw.append(1.0 + lpw1[k] - lpw2[k])
            else:
                bsw.append(lpw1[k] - lpw2[k])
        
        if plot:
            plt.plot(bsw)
            plt.title("Band Stop Filter weights")
            plt.xlabel("t")
            plt.gcf().canvas.set_window_title("Band Stop Filter weights")
            plt.show()
            
        return bsw