import numpy as np
import time

a = np.random.rand(10000000)
b = np.random.rand(10000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print(str((toc-tic)*1000)+'ms')

d = np.zeros((5,1))
print(d)