import numpy as np
from scipy.io import wavfile


class Data:
    # N - max t-argument's value for x(t)-function
    def __init__(self, N):
        self.N = N
        self.t = [*range(0, self.N)]
     
     
    def display(self):
        print(self.t)
        
    
    # Read data from file
    def read_file(self, path, dt=0.002, ext='dat', start=0, end=-1):
        if ext == 'dat':
            with open(path,'rb') as file:
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
        
    
    def write_to_wav(self, output, rate, input):
        wavfile.write(output, rate, input.astype(np.int16))