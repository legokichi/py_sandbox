import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math
import time

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024*8
RECORD_SECONDS = 60
audio = pyaudio.PyAudio()
 
# start
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print("start")
chunks = []
prev = time.time()
for i in range(0, 10):
    now = time.time()
    data = stream.read(CHUNK)
    print(type(data), len(data), now - prev)
    prev = now
    _data = np.frombuffer(data, dtype=np.int16)
    chunks.append(_data)


# stop
stream.stop_stream()
stream.close()
audio.terminate()

DATA = np.concatenate(chunks)

# 波形表示
SAMPLING = RATE
n = len(DATA)

plt.plot(DATA)
plt.show()

# スペクトル対数表示
m = math.ceil((n+1)/2.0)
p = sp.fft(DATA)
m = math.ceil((n+1)/2.0)
p=p[0:m]
p=abs(p)
p=p/float(n)
p=p**2
p[1:len(p)-1]=p[1:len(p)-1]*2
freq=sp.arange(0,m, 1.0)* (SAMPLING / n )
plt.plot(freq/1000, 10 * sp.log10(p))
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power (dB)')
plt.show()

# スペクトログラム
nFFT=1024
window=sp.hamming(nFFT)
Pxx,freqs, bins, im = plt.specgram(DATA,
                                   NFFT=nFFT, Fs=44100,
                                   noverlap=900, window=window)
plt.show()

