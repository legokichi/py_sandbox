# http://kaiseki-web.lhd.nifs.ac.jp/wiki/index.php/Python_によるオーディオ処理
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math

SAMPLING = 44.1e3
DATA = [math.sin(x/441.0*x/441.0) for x in xrange(44100)]
n = len(DATA)

plt.plot(xrange(len(DATA)), DATA)
plt.show()

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

nFFT=1024
window=sp.hamming(nFFT)
Pxx,freqs, bins, im = plt.specgram(DATA,
                                   NFFT=nFFT, Fs=44100,
                                   noverlap=900, window=window)
plt.show()
