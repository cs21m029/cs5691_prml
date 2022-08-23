# Assignment 1.7a_2

# Importing the necessary libraries
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import pandas as pd
import math

xs = np.linspace(0,5,2000)
xs1 = xs.reshape(xs.shape[0], 1)
#print(xs1)
print(xs1.shape)

ys = np.array([-1.0]*xs1.shape[0])
for i in range(xs1.shape[0]):
    ys[i] = xs1[i]/(math.exp(2*xs1[i]))
    
ys1 = ys.reshape(xs.shape[0], 1)
#print(ys1)
print(ys1.shape)

plt.plot(xs1,ys1,color="green")
plt.xlabel(r'$\theta$')
plt.ylabel("p(2|"r'$\theta$'")")
plt.title("p(x|"r'$\theta$'")  vs  "r'$\theta$'"  for  x = 2")
plt.show()
