# coding: utf-8
import numpy as np
# NumPy配列のブールインデックス参照
# https://hydrocul.github.io/wiki/numpy/ndarray-ref-boolean.html

print("numpy")
lst = np.linspace(-1.0, 1.0, 10) # http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linspace.html
print(lst)
print("lst[lst<0]", lst[lst<0])
print(len(lst))
print(len(lst[lst<0]))


print("native list")
lst = [float(i)*0.1-1.0 for i in range(10)]
print(lst)
print("lst[lst<0]", lst[lst<0])
print(len(lst))
#print(len(lst[lst<0])) # error!


print("boolean index")
a = np.array([0, 10, 20, 30, 40, 50])
b = np.array([True, False, True, False, True, False])
print(a)
print(b)
print(a[b])
