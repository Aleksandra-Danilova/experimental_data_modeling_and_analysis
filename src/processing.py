from src.analysis import Analysis
from math import pi, sin, cos, log
from matplotlib import pyplot as plt
import numpy as np
import cv2

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
            #plt.gcf().canvas.set_window_title("Potter Low Pass Filter Inverse Fourier Transform (LPF weights)")
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
            #plt.gcf().canvas.set_window_title("High Pass Filter weights")
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
            #plt.gcf().canvas.set_window_title("Band Pass Filter weights")
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
            #plt.gcf().canvas.set_window_title("Band Stop Filter weights")
            plt.show()
            
        return bsw
    
    
    # Gradational transformations
    # Get negative image with s = L - 1 -r,
    # where r and s are pixels of original and transformed images respectively,
    # L - maximum value of original image.
    def get_negative(self, image):
        const = np.amax(image) - 1
        negative = np.zeros(image.shape, dtype=int)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                negative[i][j] = int(const - image[i][j])
        return negative
    
    
    # Gamma correction: s = C * r^y, where C > 0, y > 0
    def gamma_conversion(self, image, C, gamma):
        gamma_corrected = np.zeros(image.shape)
        if C > 0 and gamma > 0:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    gamma_corrected[i][j] = C * ((image[i][j])**gamma)
        return gamma_corrected
    
    
    # Logarithmic transformation: s = C * log(r + 1), where C > 0
    def logarithmic_transformation(self, image, C):
        log_transform = np.zeros(image.shape)
        if C > 0:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    log_transform[i][j] = C * log(image[i][j] + 1)
        return log_transform
    
    
    # Gradation method of MxN size by histogram equalization
    def normalized_histogram(self, image):
        # normalized histogram p(r_k) = n_k / MN of the source image,
        # where k is level of brightness in range [0, L],
        # L is a maximum brightness value in original image,
        # r is a pixel of the source image.
        
        L = 255
        print(image)
        p = np.zeros(L + 1)
        #size_mult = 1 / (image.shape[0] * image.shape[1])
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                p[image[i][j]] += 1
        
        MN = image.shape[0] * image.shape[1]
        p = p / MN #[count * size_mult for count in p]
        
        plt.plot()
        plt.clf()
        plt.plot(p)
        plt.title("Histogram")
        plt.xlabel("r_k")
        plt.ylabel("p(r_k)")
        plt.show()
        
        return p
    
    
    def CDF(self, image, hist, show=True):
        S = 255
        cdf = np.zeros(S + 1)
        
        for i in range(len(hist)):
            for j in range(0, i + 1):
                cdf[i] += hist[j]
        
        if show:
            plt.plot()
            plt.clf()
            plt.plot(cdf)
            plt.title("CDF")
            plt.show() 
        
        return cdf
    
    
    def histogram_equalization(self, image, cdf, show=True):
        L = np.max(image)
        equalized = np.zeros(image.shape, dtype=int)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                equalized[i][j] = int(cdf[image[i][j]] * L)
        
        print(equalized)
        if show:
            plt.imshow(equalized, cmap='gray')
            plt.title("Histogram Equalized")
            plt.show()
            
        return equalized
    
    
    def detect_artefacts(self, image, axis=1, show=True):
        M = 10
        plt.imshow(image, cmap='gray')
        plt.title("xcr")
        plt.show()
        
        N = image.shape[1]
        
        if axis == 0:
            for row in range(0, 52, 10):
                Xn = Analysis(N, M).Fourier(image[row], N)
                fourier = Analysis(N, M).spectrFourier(Xn, N, image[row], dt=1)
    
            for row in range(0, 52, 10):
                derivative = list()
                for i in range(0, N-1):
                    derivative.append((image[row][i] - image[row][i + 1]))
                derivative.append(derivative[-1])    

                fig, axis = plt.subplots(1, 2, figsize=(8, 4), num="Data & Derivative")
                fig.suptitle("Data & Derivative")
                axis[0].plot(image[row])
                axis[1].plot(derivative)
                for ax in axis.flat:
                    ax.set(xlabel='t', ylabel='x(t)')
                fig.tight_layout()  
                plt.show()
      
                Xn = Analysis(N, M).Fourier(derivative, N)
                fourier = Analysis(N, M).spectrFourier(Xn, N, derivative, dt=1)
        
                acf = Analysis(N, M).acf(derivative, N)
                Xn_acf = Analysis(N, M).Fourier(acf, N)
                fourier = Analysis(N, M).spectrFourier(Xn_acf, N, acf, dt=1)
        
        elif axis == 1:
            for col in range(0, 52, 10):
                column = [row[col] for row in image]
                Xn = Analysis(N, M).Fourier(column, N)
                fourier = Analysis(N, M).spectrFourier(Xn, N, column, dt=1)

            
            for col in range(0, 52, 10):
                if col != 0:
                    derivative_previous = derivative
                derivative = list()
                column = [row[col] for row in image]
                
                for i in range(0, N-1):
                    derivative.append((column[i] - column[i + 1]))
                derivative.append(derivative[-1])    

                if col == 0:
                    derivative_previous = derivative
                    
                fig, axis = plt.subplots(1, 2, figsize=(8, 4), num="Data & Derivative")
                fig.suptitle("Data & Derivative")
                axis[0].plot(column)
                axis[1].plot(derivative)
                for ax in axis.flat:
                    ax.set(xlabel='t', ylabel='x(t)')
                fig.tight_layout()  
                plt.show()
      
                Xn = Analysis(N, M).Fourier(derivative, N)
                fourier = Analysis(N, M).spectrFourier(Xn, N, derivative, dt=1)
        
                acf = Analysis(N, M).acf(derivative, N)
                Xn_acf = Analysis(N, M).Fourier(acf, N)
                fourier = Analysis(N, M).spectrFourier(Xn_acf, N, acf, dt=1)
                
                ccf = Analysis(N, M).ccf(derivative_previous, derivative, N)
                Xn_ccf = Analysis(N, M).Fourier(ccf, N)
                fourier = Analysis(N, M).spectrFourier(Xn_ccf, N, ccf, dt=1)
                
        return 0
    
    
    
    def convolution(self, xt, ht_norm, N=1000, M=200, dt=0.005, plot=True):
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
    
    
    def suppress_artifacts(self, image, fc1=0.25, fc2=0.5, m=32):
        conv = list()
        N = image.shape[0]
        
        for row in image:
            spectrum = row
            bsf = Processing(N).bsf(fc1=fc1, fc2=fc2, dt=1, m=m, plot=False)
            conv_bsf = self.convolution(spectrum, bsf, N=N, M=2*m+1, dt=1, plot=False)                 
            conv.append(conv_bsf[2*m+1:-1])

        plt.imshow(conv, cmap='gray')
        plt.title("xcr without artifacts")
        plt.show()
        
    
    
    def __median_filter(self, image, filter_size):
        temp = list()
        N = image.shape[0]
        M = image.shape[1]
        indexer = filter_size // 2
        data_final = np.zeros((N, M))
        
        for i in range(N):
            for j in range(M):
                for z in range(filter_size):
                    if i + z - indexer < 0 or i + z - indexer > N - 1:
                        for c in range(filter_size):
                            temp.append(0)
                    else:
                        if j + z - indexer < 0 or j + indexer > M - 1:
                            temp.append(0)
                        else:
                            for k in range(filter_size):
                                temp.append(image[i + z - indexer][j + k - indexer])

                temp.sort()
                data_final[i][j] = temp[len(temp) // 2]
                temp = []
        return data_final

    
    def __arithmetic_mean_filter(self, image, filter_size):
        N = image.shape[0]
        M = image.shape[1]
        indexer = filter_size // 2
        output = np.zeros(image.shape, np.uint8)
        
        
        for i in range(N):
            for j in range(M):
                try:
                    box = image[i-indexer:i+indexer, j-indexer:j+indexer]
                    mean_value = np.mean(box)
                    output[i, j] = mean_value
                except:
                    output[i, j] = 0
                
        return output
    
   
    # Method to suppress additive noise on modeled images
    # using different spatial filters with masks of different sizes
    def suppress_noise(self, image, filter_size=3, filter_type='median'):
        if filter_type == 'median':
            filtered = self.__median_filter(image, filter_size)
            plt.imshow(filtered, cmap='gray')
            plt.title(f"Median Filter, size={filter_size}")
            plt.show()
            return filtered
        elif filter_type == 'mean':
            filtered = self.__arithmetic_mean_filter(image, filter_size)
            plt.imshow(filtered, cmap='gray')
            plt.title(f"Arithmetic Mean Filter, size={filter_size}")
            plt.show()
            return filtered
            
            
    
    def inverse_filter(self, data1, data2, N=1000, M=200, alpha=0):    
        X_hat = list()
        alpha_sqr = alpha**2

        # Use formulas for complex numbers division:
        # Re[Xk_hat] = (Re[Yk] * Re[Hk] + Im[Yk] * Im[Hk]) / (Re[Hk]^2 + Im[Hk]^2), and
        # Im[Xk_hat] = (Im[Yk] * Re[Hk] - Re[Yk] * Im[Hk]) / (Re[Hk]^2 + Im[Hk]^2).
        Re = [(data1[0][i] * data2[0][i] + data1[1][i] * data2[1][i]) / (data2[0][i]**2 + data2[1][i]**2 + alpha_sqr) for i in range(N)]
        Im = [(data1[1][i] * data2[0][i] - data1[0][i] * data2[1][i]) / (data2[0][i]**2 + data2[1][i]**2 + alpha_sqr) for i in range(N)]

        X_hat = [Re[i] + Im[i] for i in range(N)]        
        inverse_fourier = Analysis(N, M).inverse_Fourier(X_hat, len(X_hat), include_complex=False, show=False)
        return inverse_fourier
            
        
    
    def enlarge_data(self, data, coefficient):
        N = len(data)
        Xn = Analysis(N, N).Fourier(data, N)
        tail = int(N * (coefficient / 2) - N // 2)
        zeros = [0] * tail
        fourier_with_zeroes = Xn + zeros
        
        plt.plot(fourier_with_zeroes)
        plt.show()
        
        full_spectrum = fourier_with_zeroes[::-1] + fourier_with_zeroes
        return full_spectrum
    
    
    def complex_spectrum(self, data, N, L=0, noise=False, next_division=False):
        Re = list()
        Im = list()
        const_2pi_N = 2 * pi / N
        
        # set to 0 the last L values
        if L > 0:
            for i in range(N-L, N):
                data[i] = 0
                
        for n in range(N):
            Re_n = 0
            Im_n = 0
            for k in range(N):
                xk = data[k]
                Re_n += xk * cos(const_2pi_N*n*k)
                Im_n += xk * sin(const_2pi_N*n*k)
            Re.append(Re_n / N)
            Im.append(Im_n / N)
                
        # complex spectrum
        if noise:
            Xn_complex = [Re[i] + (-Im[i]) for i in range(N)]
            return Xn_complex
        elif noise == False and next_division == True:
            return Re, Im
        else:
            Xn_complex = [Re[i] + Im[i] for i in range(N)]
            return Xn_complex
                
    
    
    def binarize_image(self, image, threshold_percantage):
        threshold_value = threshold_percantage * np.max(image)
        image[image < threshold_value] = 0
        image[image >= threshold_value] = 255
        
        return image
    
    
    '''
        Edge detection by differential operators:
        gradient (Prewitt and Sobel).
        Precise calculation: G = (G_x^2 + G_y^2)^(1/2)
    '''
    def __egde_detection(self, image, masks, mask_type):
        if mask_type == 0:
            operator_x = masks[0]
            operator_y = masks[1]    
        elif mask_type == 1:
            operator_x = masks[2]
            operator_y = masks[3]
            
        G = np.array(image)
        
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                G_x = np.sum(image[i - 1 : i + 2, j - 1 : j + 2] * operator_x)
                G_y = np.sum(image[i - 1 : i + 2, j - 1 : j + 2] * operator_y)
                G[i, j] = np.sqrt(G_x**2 + G_y**2)
        return G
        

    def sobel_operator(self, img, threshold=0.5):
        new_image = np.copy(img)
        size = img.shape
        for i in range(1, size[0] - 1):
            for j in range(1, size[1] - 1):
                G_x = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
                G_y = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
                new_image[i][j] = np.sqrt(G_x**2 + G_y**2)
       

        plt.imshow(new_image, cmap='gray')
        plt.title("Sobel Operator")
        plt.show()

            
        binary_image = Processing(100).binarize_image(new_image, threshold)
        plt.imshow(binary_image, cmap='gray')
        plt.title('Edges after Sobel operator')
        plt.show()
        
        return new_image
    
   
    
    
    def prewitt_operator(self, image, mask_type=0, threshold=0.6, show=True):
        prewitt_masks = [
            [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], # applied to image's X axis
            [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], # applied to image's Y axis
            [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], # applied to image's X axis
            [[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]] # applied to image's Y axis

        G = self.__egde_detection(image, prewitt_masks, mask_type)
                
        if show:        
            plt.imshow(G, cmap='gray')
            plt.title("Prewitt Operator")
            plt.show()
            
            binary_image = Processing(100).binarize_image(G, threshold)
            plt.imshow(binary_image, cmap='gray')
            plt.title('Edges after Prewitt operator')
            plt.show()
        
        return binary_image
    
    def apply_filter(self, img, f):
        img_h, img_w = img.shape
        ret = np.array(img, dtype='float')
        img = np.array(img, dtype='float')
        f_h, f_w = f.shape
        assert f_h % 2 == 1, 'assume filter size is odd'
        f_size = np.int((f_h - 1) / 2)

        for i in range(img_h):
            for j in range(img_w):
                if (i - f_size < 0 or j - f_size < 0 
                    or i + f_size >= img_h or j + f_size >= img_w):
                    ret[i][j] = 0
                    continue
                v = 0
                for di in range(-f_size, f_size + 1):
                    for dj in range(-f_size, f_size + 1):
                        ci = i + di
                        cj = j + dj
                        fi = di + f_size
                        fj = dj + f_size
                        v = v + f[fi, fj] * img[ci, cj]
                ret[i][j] = v

        return ret


        
    
    
    def erosion(self, image, shape):
        image_1 = np.array(image)
        image_2 = np.array(image)
        
        mx = shape[0] // 2
        my = shape[1] // 2
        
        kernel = np.ones((shape[0], shape[1])) * 255
        
        for i in range(mx, image.shape[0] - mx):
            for j in range(my, image.shape[1] - my):
                tt = image_1[i - mx : i + mx + 1, j - my : j + my + 1] * kernel
                if np.min(tt) == 0:
                    image_2[i - mx : i + mx + 1, j - my : j + my + 1] = 0
        plt.imshow(image_2, cmap='gray')
        plt.title('Erosion (mask size = ' + str(shape[0]) + ')')
        plt.show()            
        return image_2
    
    
    def dilation(self, image, shape):
        image_1 = np.array(image)
        image_2 = np.array(image)
        
        mx = shape[0] // 2
        my = shape[1] // 2
        
        kernel = np.ones((shape[0], shape[1])) * 255
        for i in range(mx, image.shape[0] - mx):
            for j in range(my, image.shape[1] - my):
                tt = image_1[i - mx : i + mx + 1, j - my : j + my + 1] * kernel
                if np.sum(tt) > 0:
                    image_2[i - mx : i + mx + 1, j - my : j + my + 1] = 255
        plt.imshow(image_2, cmap='gray')
        plt.title('Dilation (mask size = ' + str(shape[0]) + ')')
        plt.show()            
        return image_2