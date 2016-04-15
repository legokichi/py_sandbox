# coding: utf-8
import sys
import numpy as np
import matplotlib.pylab as plt
import mtxplot

print sys.version

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

'''
np.sin(np.array((0., 30., 45., 60., 90.)) * np.pi / 180. )
array([ 0.        ,  0.5       ,  0.70710678,  0.8660254 ,  1.        ])
Plot the sine function:
'''
