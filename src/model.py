import cv2
import random
from src.data import Data
import numpy as np
from math import sin, pi
from matplotlib import pyplot as plt, image

   
class Model(Data):
    def __init__(self, N):
        super(Model, self).__init__(N)
        
       
    # function x(t) = a * t + b
    def _generate_linear(self, a, b):
        x_linear = [a * element + b for element in self.t]
        x_linear2 = [-a * element + b for element in self.t]     
        return x_linear, x_linear2
    
    
    # function x(t) = b * exp(-alpha * t)
    def _generate_exp(self, b, alpha):
        if (alpha == 0 or b == 0):
            print ("Parameters alpha and b cannot equal 0")
        x_exp = [b * np.exp(alpha * element) for element in self.t]
        x_exp2 = [b * np.exp(-alpha * element) for element in self.t]
        return x_exp, x_exp2
        
        
    def _generate_piecewise(self, a, b, alpha):
        dot1 = random.randint(int(0.1*self.N), int(0.3*self.N))
        dot2 = dot1 + int(0.5*self.N)
        
        x_piece1 = [a * element + b for element in self.t if (element <= dot1)]
        x_piece2 = [b * np.exp(alpha * element) for element in self.t if (element > dot1 and element <= dot2)]
        x_piece3 = [-a * element + b for element in self.t if (element > dot2)]
        x_piecewise = x_piece1 + x_piece2 + x_piece3
        return x_piecewise
    
        
    def trend(self, a, b, alpha, plot_graphs=True):
        graphs = list()
        if (a == 0 or b == 0 or alpha == 0):
            print ("Parameters a, b or alpha cannot equal 0")
        if plot_graphs == True:
            fig_linear, axis_linear = plt.subplots(1, 2, figsize=(5, 3), num="Linear Functions")
            fig_linear.suptitle("Linear Functions")
            axis_linear[0].plot(self.t, self._generate_linear(a, b)[0])
            axis_linear[1].plot(self.t, self._generate_linear(a, b)[1])
            for ax in axis_linear.flat:
                ax.set(xlabel='t', ylabel='x(t)')
            fig_linear.tight_layout()
        
            fig_exp, axis_exp = plt.subplots(1, 2, figsize=(5, 3), num="Exponential Functions")
            fig_exp.suptitle("Exponential Functions")
            axis_exp[0].plot(self.t, self._generate_exp(b, alpha)[0])
            axis_exp[1].plot(self.t, self._generate_exp(b, alpha)[1])
            for ax in axis_exp.flat:
                ax.set(xlabel='t', ylabel='x(t)')
            fig_exp.tight_layout()
            
            fig_piece, axis_piece = plt.subplots(1, 1, figsize=(5, 4), num="Piecewise Function")
            fig_piece.suptitle("Piecewise Function")
            axis_piece.plot(self.t, self._generate_piecewise(a, b, alpha))
            axis_piece.set(xlabel='t', ylabel='x(t)')
            
            plt.show()          
                        
        graphs.append(self._generate_linear(a, b)[0])
        graphs.append(self._generate_linear(a, b)[1])
        graphs.append(self._generate_exp(b, alpha)[0])
        graphs.append(self._generate_exp(b, alpha)[1])
        graphs.append(self._generate_piecewise(a, b, alpha))               
                    
        return graphs
        
            
    def __recalculate(self, xk, R):
        xk_min = min(xk)
        xk_max = max(xk)
        xk_hat = [((element - xk_min) / (xk_max - xk_min) - 0.5) * 2 * R for element in xk]
        return xk_hat
    
        
    def __draw_plot(self, xk_hat, type):
        plt.plot(xk_hat)
        plt.title(f"{type}")
        plt.xlabel("t")
        plt.ylabel("x(t)")
        #if type == "Noise":
        #    plt.gcf().canvas.set_window_title("Noise")
        #elif type == "Customer Noise":
        #    plt.gcf().canvas.set_window_title("Customer Noise")
        plt.show()
        
        
    def noise(self, N, R, plot_graphs=True):
        # generate random noise
        xk = list()
        for i in range(0, N):
            random_number = random.randint(0, N)
            xk.append(random_number)
        
        xk_hat = list()
        if R > 0:
            xk_hat = self.__recalculate(xk, R)
            if plot_graphs:
                self.__draw_plot(xk_hat, "Noise")
        else:
            print("The noise parameter R parameter must meet the condition: R > 1.")
        return xk_hat
        
            
    def customer_noise(self, N, R, plot_graphs=True):
        xk = list()
        for i in range(0, N):
            prev = i
            random_number = ((i * 0.1 % 5) + prev % 23)
            xk.append(random_number)

        xk_hat = list()
        if R > 0:
            xk_hat = self.__recalculate(xk, R)
            if plot_graphs:
                self.__draw_plot(xk_hat, "Customer Noise")
        else:
            print("The customer noise parameter R must meet the condition: R > 0.")
        
        return xk_hat
                    
                    
    # shift input data by constant                
    def shift(self, inputData, C=0, N1=0, N2=-1, plot_graphs=True):
        shifted_data = list()
        
        if N1 == 0 and N2 == -1:
            shifted_data = [inputData[t] + C for t in range(len(inputData))]
        elif N1 <= N2:
            shifted_data = [inputData[t] + C if t >= N1 and t <= N2 else inputData[t] for t in range(len(inputData))]              
        
        if plot_graphs == True:
            fig, axis = plt.subplots(1, 2, figsize=(5, 3), num="Original & Shifted Functions")
            fig.suptitle("Original & Shifted Functions")
            axis[0].plot(self.t, inputData)
            axis[1].plot(self.t, shifted_data)
            for ax in axis.flat:
                ax.set(xlabel='t', ylabel='x(t)')
            fig.tight_layout()  
            plt.show()  
              
        return shifted_data
    
    
    # shift input 2D-data by constant                
    def shift_2D(self, data, C=30, new_filename='shifted.jpg', write=True, show=True):
        shifted_data = list()

        # add a constant to each element in rows
        for row in range(len(data[0])):
            temp_row = list()
            for col in range(len(data[0][row])):
                temp_row.append(data[0][row][col] + C)
            shifted_data.append(temp_row)
        # convert to numpy array
        shifted_data = np.array(shifted_data)
        print(np.amin(shifted_data), np.amax(shifted_data))
        
        # write array to a file to save as an image
        if write and show:        
            cv2.imwrite(new_filename, shifted_data)
            # read the image and show it
            if show:
                img = cv2.imread(new_filename, cv2.IMREAD_GRAYSCALE)
                plt.imshow(img, cmap='gray')
                plt.title("Shifted")
                plt.show()
            
        return shifted_data
    
    
    # generates M outliers
    def impulse_noise(self, data, N, M, R, type='template', plot=False):
        #if not (M >= int(0.005 * N) and M <= int(0.01 * N)):
            #M = random.randint(int(0.005 * N), int(0.01 * N))
        #    M = int(input('M = '))
        #    print(f'Parameter M is not relevant. Updated M is {M}')
        Rs = 0.1 * R   
        outliers = dict()
        impulse_plot = list()
        
        for outlier in range(0, M):
            t = random.randint(0, N) 
                        
            sign = "+-"
            xt = R
            
            xt_sign = random.choice(sign)
            if xt_sign == '-':
                xt = -1 * xt
            
            subrange_0 = random.randint(int(R-Rs), int(R+Rs))
            subrange_1 = random.randint(int(-R-Rs), int(-R+Rs))

            if xt > 0:
                xt = subrange_0
            else:
                xt = subrange_1
            
            outliers[t] = xt
        
        for i in range(N):
            if (i in outliers):
                impulse_plot.append(outliers[i])
            else:
                if (type == 'template'):
                    impulse_plot.append(0)
                elif (type == 'data'):
                    impulse_plot.append(data[i])
                
        if plot:
            plt.clf()
            plt.plot([n * 0.00071 for n in range(N)], impulse_plot)
            plt.title("Impulse Noise")
            plt.xlabel("t [s]")
            plt.ylabel("x(t)")
            plt.gcf().canvas.set_window_title("Impulse Noise")
            plt.show()
                
        return impulse_plot
    
    
    # harmonic process
    def harmonic(self, N, A0, f0, delta_t, update=True, plot_graphs=True):
        const_2pi_delta_t = 2 * pi * delta_t
        xt = [A0 * sin(const_2pi_delta_t*f0*k) for k in range(N)]
        if plot_graphs:
            plt.plot([n * delta_t for n in range(N)], xt)
            plt.title(f"Harmonic Process (f0={f0} [Hz])")
            plt.xlabel("t [s]")
            plt.ylabel("x(t)")
            plt.gcf().canvas.set_window_title("Harmonic Process")
            plt.show()
        
        if update:
            i = 1
            start = f0 + 50
            if plot_graphs:
                for step in range(start, 533+1, 50):
                    plt.plot([A0 * sin(const_2pi_delta_t*step*k) for k in range(N)])
                    plt.title(f"Harmonic Process (f0={step} [Hz])")
                    plt.xlabel("t")
                    plt.ylabel("x(t)")
                    plt.gcf().canvas.set_window_title(f"Harmonic Process {i}")
                    i += 1
                    plt.show()
        
        return xt
    
    
    # polyharmonic process
    def polyHarm(self, N, Ai, fi, delta_t, plot_graph=True):
        const_2pi = 2 *pi
        xt = list()
        if delta_t > 1 / (2 * fi[2]):
            print("Parameter delta_t must be <= 1/2f2.")
            delta_t = input()
        for k in range(N): 
            xk = 0
            for i in range(3):
                xk += Ai[i] * sin(const_2pi*fi[i]*delta_t*k)
            xt.append(xk)
        
        if plot_graph == True:
            plt.plot(xt)
            plt.title("Polyharmonic Process")
            plt.xlabel("t")
            plt.ylabel("x(t)")
            plt.gcf().canvas.set_window_title("Polyharmonic Process")
            plt.show()
        
        return xt
    
    
    # additive model for elementwise addition
    def addModel(self, data1, data2, N, plot=True):
        data = list()
        for k in range(0, N):
            data.append(data1[k] + data2[k])
        
        if plot:
            fig, axis = plt.subplots(1, 3, figsize=(10, 4), num="Original Functions & Additive Model")
            fig.suptitle("Original Functions & Additive Model")
            axis[0].plot(self.t, data1)
            axis[1].plot(self.t, data2)
            axis[2].plot(self.t, data)
            
            for ax in axis.flat:
                ax.set(xlabel='t', ylabel='x(t)')
            fig.tight_layout()  
            plt.show()  
        
        return data
    
    
    
    # additive model for elementwise multiplication
    def multModel(self, data1, data2, N, plot_graphs=True, x_axis=None):
        data = list()
        for k in range(0, N):
            data.append(data1[k] * data2[k])
        
        if x_axis is None:
            x_axis = self.t
            
        if plot_graphs == True:
            fig, axis = plt.subplots(1, 3, figsize=(10, 4), num="Original Functions & Multiplicative Model")
            fig.suptitle("Original Functions & Multiplicative Model")
            axis[0].plot(x_axis, data1)
            axis[1].plot(x_axis, data2)
            axis[2].plot(x_axis, data)
            
            for ax in axis.flat:
                ax.set(xlabel='t', ylabel='x(t)')
            fig.tight_layout()  
            plt.show()  
        
        return data
    
    
    # additive model for elementwise multiplication of 2D-array
    def multModel_2D(self, data, C=1.3, new_filename='multiplied.jpg', write=True, show=True):
        multiplied_by_const = list()

        # add a constant to each element in rows
        for row in range(len(data[0])):
            temp_row = list()
            for col in range(len(data[0][row])):
                temp_row.append(data[0][row][col] * C)
            multiplied_by_const.append(temp_row)
        # convert to numpy array
        multiplied_by_const = np.array(multiplied_by_const)
        print(np.amin(multiplied_by_const), np.amax(multiplied_by_const))
        
        # write array to a file to save as an image
        if write:     
            cv2.imwrite(new_filename, multiplied_by_const)
            # read the image and show it
            if show:
                img = cv2.imread(new_filename, cv2.IMREAD_GRAYSCALE)
                plt.imshow(img, cmap='gray')
                plt.title("Multiplied")
                plt.show()
        return multiplied_by_const
    
    
    
    # Method to create noise of different kinds on image
    def noise_image(self, image, noise_type='salt_pepper', M=2, R_imp=255, R_rnd=100):
        N = image.shape[1]
        additive_model = list()
        
        if noise_type == 'salt_pepper':
            for row in range(0,  image.shape[0]):
                noise = self.impulse_noise(image[row], N, M=M, R=R_imp, plot=False)
                add = self.addModel(image[row], noise, N, plot=False)
                additive_model.append(add)        
            plt.imshow(additive_model, cmap='gray')
            plt.title("Salt and Pepper")
            plt.show()  
          
        elif noise_type == 'random':
            for row in range(0,  image.shape[0]):
                noise = self.noise(N, R=R_rnd, plot_graphs=False)
                add = self.addModel(image[row], noise, N, plot=False)
                additive_model.append(add)        
            plt.imshow(additive_model, cmap='gray')
            plt.title("Random")
            plt.show()  
        
        elif noise_type == 'both':
            for row in range(0, image.shape[0]):
                imp_noise = self.impulse_noise(image[row], N, M=M, R=R_imp, plot=False)
                rnd_noise = self.noise(N, R=R_rnd, plot_graphs=False)
                noise = self.addModel(imp_noise, rnd_noise, N, plot=False)
                add = self.addModel(image[row], noise, N, plot=False)
                additive_model.append(add)        
            plt.imshow(additive_model, cmap='gray')
            plt.title("Both")
            plt.show()  
        
        return additive_model