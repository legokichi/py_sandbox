# coding: utf-8
import numpy as np
import numba
from numba import jit, complex128
import matplotlib.pyplot as plt

size = 200
iterations = 100

# @jit decorator
@jit(locals=dict(c=complex128, z=complex128))
def mandelbrot(m, size, iterations):
    for i in range(size):
        for j in range (size):
            c = -2 + 3.0/size*j + 1j*(1.5 - 3.0/size*i)
            z = 0
            for n in range(iterations):
                if np.abs(z) <= 10:
                    z = z*z + c
                    m[i, j] = n
                else:
                    break

m = np.zeros((size, size))
mandelbrot(m, size, iterations)
plt.imshow(np.log(m), cmap=plt.cm.hot)
plt.xticks([])
plt.yticks([])
plt.show()
