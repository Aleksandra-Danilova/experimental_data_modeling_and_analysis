import numpy as np
from scipy.io import wavfile
import cv2
import matplotlib.pyplot as plt


class Data:
    # N - max t-argument's value for x(t)-function
    def __init__(self, N):
        self.N = N
        self.t = [*range(0, self.N)]
     
     
    def display(self):
        print(self.t)
        
    
    # Read data from file
    def read_file(self, path, dt=0.002, ext='dat', start=0, end=-1, show=True):
        if ext == 'dat':
            with open(path, 'rb') as file:
                # convert bytes to floats
                bytes = bytearray(file.read())
                floats = list(np.frombuffer(bytes, dtype=np.float32)) # ordinate
                t = list([i * dt for i in range(len(floats))]) # abscissa (seconds)
            return t, floats
        
        elif ext == 'wav':
            # Read *.wav file
            # and display its sampling rate value and length of record
            rate, data = wavfile.read(path)
            audio_fragment = data[start:end]
            N = len(audio_fragment)
            time = audio_fragment.shape[0] / rate
            print('Rate =', rate, '[Hz]', '\nLength =', N, '\nTime length =', time, '[s]')    
            time_linspace = np.linspace(0., time, audio_fragment.shape[0])
            return time_linspace, audio_fragment, rate, N
        
        elif ext == 'jpg':
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if show:
                plt.imshow(img, cmap=plt.get_cmap('gray'))# cmap='gray')
                plt.title("Original")
                plt.show()
            return img, img.size
          
        
    def write_to_wav(self, output, rate, input):
        wavfile.write(output, rate, input.astype(np.int16))
        
        
    # Bringing the data of any image to a gray scale, i.e. from an arbitrary range of values 
    # to a range of S = [0, 255] according to the formula xk_hat = (xk - x_min) / (x_max - x_min) * S,
    # where xk - values of the original image, xk_hat - values of the reduced image, S = 255.
    def recalculate_to_grayscale(self, image, 
                                 new_filename='recalculated.jpg', 
                                 write=True, show=True):
        img = image
        x_min = np.amin(img)
        x_max = np.amax(img)
        s = 255
        recalculated_list = list()
        
        for row in range(len(img)):
            temp_row = list()
            for col in range(len(img[row])):
                temp_row.append(((img[row][col] - x_min) / (x_max - x_min)) * s)
            recalculated_list.append(temp_row)
        recalculated_list = np.array(recalculated_list)

        print(np.amin(recalculated_list), np.amax(recalculated_list))
        
        # write array to a file to save as an image
        if write:
            cv2.imwrite(new_filename, recalculated_list)
            # read the image and show it
            if show:
                img = cv2.imread(new_filename, cv2.IMREAD_GRAYSCALE)
                plt.imshow(img, cmap='gray')
                plt.title("Recalculated to Grayscale")
                plt.show()
        return recalculated_list
