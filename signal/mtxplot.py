# coding: utf-8
import sys
print sys.version

import numpy as np
import matplotlib.pylab as plt

def plot(fnmtx):
    w = len(fnmtx[0])
    h = len(fnmtx)
    k = 1
    for fnarr in fnmtx:
        for fn in fnarr:
            plt.subplot(w,h,k)
            fn(k)
            k += 1
'''example

def plotSin(id):
    x = np.linspace(-np.pi, np.pi, 201)
    plt.plot(x, np.sin(x))
    plt.xlabel('Angle [rad]')
    plt.ylabel('sin(x)')
    plt.axis('tight')

plot([
    [
        plotSin,
        plotSin,
    ],
    [
        plotSin,
        plotSin,
    ],
])
plt.show()
'''
