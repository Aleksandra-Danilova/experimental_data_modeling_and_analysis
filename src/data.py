import numpy as np
from scipy.io import wavfile
import cv2
import math
import matplotlib.pyplot as plt

class Data:
    # N - max t-argument's value for x(t)-function
    def __init__(self, N):
        self.N = N
        self.t = [*range(0, self.N)]
     
     
    def display(self):
        print(self.t)
        
    
    # Read data from file
    def read_file(self, path, dt=0.002, ext='dat', start=0, end=-1, shape=(1024, 1024), show=True, rot=False):
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
            #cv2.imshow("Original", img)
            
            if show:
                plt.imshow(img, cmap=plt.get_cmap('gray'))#, vmin=0, vmax=255)
                plt.title("Original")
                plt.show()
            return img, img.size
        
        elif ext == 'xcr':
            with open(path, "rb") as f:
                # header of start bytes may be a text, ignore tail of end bytes
                bytes = bytearray(f.read())[start:end]
                
                # after the header, two-byte image data comes
                data_to_hex = bytes.hex()
                
                data_to_hex_list = list()
                for i in range(0, len(data_to_hex), 4):
                    tetrad = data_to_hex[i] + data_to_hex[i + 1] + \
                             data_to_hex[i + 2] + data_to_hex[i + 3]
                    data_to_hex_list.append(tetrad)
                
                data_int = list()
                for i in range(len(data_to_hex_list)):
                    data_int.append(int(data_to_hex_list[i], 16))
                             
                reshaped = np.reshape(data_int, shape)
                if rot == True:
                    reshaped = np.rot90(reshaped)
                recalculate = self.recalculate_to_grayscale(reshaped, write=False, show=True)
            return recalculate

        elif ext == 'bin':
            with open(path, "rb") as f:
                bytes = np.fromfile(path, dtype='uint16')
            reshaped = np.reshape(bytes, shape)
            recalculate = self.recalculate_to_grayscale(reshaped, write=False, show=True)
            return recalculate
                
        
    def write_to_wav(self, output, rate, input):
        wavfile.write(output, rate, input.astype(np.int16))
        
        
    def write_to_xcr(self, input, path):
        with open(path, "wb") as f:
            f.write(input)
        
        
    # Bringing the data of any image to a gray scale, i.e. from an arbitrary range of values 
    # ​​to a range of S = [0, 255] according to the formula xk_hat = (xk - x_min) / (x_max - x_min) * S,
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
    
    
    
    def __nearest_neighbor_interpolation(self, image, coef): 
        new_height = int(coef * image.shape[0])
        new_width = int(coef * image.shape[1])       
        output = np.zeros((new_height, new_width))

        for i in range(new_height - 1):
            for j in range(new_width - 1):
                output[i + 1][j + 1] = image[1 + int(i / coef)][1 + int(j / coef)]

        return output
        
    
    def __bilinear_interpolation(self, image, coef):
        height, width = image.shape
        
        new_height = int(coef * height)
        new_width = int(coef * width)       
                
        image = np.pad(image, ((0, 1), (0, 1)), 'constant')
        output = np.zeros((new_height, new_width), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                scrx = (i + 1) * (height / new_height) - 1
                scry = (j + 1) * (width / new_width) - 1
                x = math.floor(scrx)
                y = math.floor(scry)
                u = scrx - x
                v = scry - y
                output[i, j] = (1 - u) * (1 - v) * image[x, y] + \
                    u * (1 - v) * image[x + 1, y] + \
                    (1 - u) * v * image[x, y + 1] + \
                    u * v * image[x + 1, y + 1]
        return output

    
    
    def resize_image(self, image, scale_factor, method='nearest_neighbor'):
        if method == 'nearest_neighbor':
            resized = self.__nearest_neighbor_interpolation(image, scale_factor)
        elif method == 'bilinear':
            resized = self.__bilinear_interpolation(image, scale_factor)
        return resized