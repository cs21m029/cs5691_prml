# Assignment 1.7a

# Importing the necessary libraries
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import pandas as pd
import math

xs = np.linspace(-10,10,2000)
xs1 = xs.reshape(xs.shape[0], 1)
#print(xs1)
print(xs1.shape)

ys = np.array([-1.0]*xs1.shape[0])
for i in range(xs1.shape[0]):
    if xs1[i] >= 0:
        ys[i] = 1/(math.exp(xs1[i]))
    else:
        ys[i] = 0

ys1 = ys.reshape(xs.shape[0], 1)
#print(ys1)
print(ys1.shape)

plt.plot(xs1,ys1,color="green")
plt.xlabel("x")
plt.ylabel("p(x|1)")
plt.title("p(x|"r'$\theta$'")  vs  "r'$\theta$'"  for  "r'$\theta$'" = 1")
plt.show()
