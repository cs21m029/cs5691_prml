# Assignment 1.2a

# Importing the necessary libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math

data = pd.read_csv("Q2_classification_test_data.csv")
dt = np.array(data)

X = dt[:, 0].reshape(dt.shape[0], 1)
Y = dt[:, 1].reshape(dt.shape[0], 1)

XY = np.append(X,Y,axis=1)

#class output of x and y
op = dt[:, 2].reshape(dt.shape[0], 1)

color = np.empty(dt.shape[0],dtype=object)

for i in range(dt.shape[0]):
    if op[i] == 1:
        color[i] = "red"
    else:
        color[i] = "green"

w1 = np.array(([0.0],[-17.49182526],[16.76442186]))
#print(w1.shape)
w = w1.reshape(w1.shape[0],1)
print(w.shape)
print(w)

min_1x = abs(X.max())
min_1y = abs(Y.max())
min_0x = min_1x
min_0y = min_1y

dist = math.sqrt(min_1x*min_1x + min_1y*min_1y)
print("Intial max Distance : ",dist)

# Y1 is Y_hat
opt = np.array([-1.0]*dt.shape[0])
op1 = opt[:].reshape(op.shape[0],1)

#print(op.shape)
#print(op1.shape)
# set Margin value
margin = 1


for i in range(dt.shape[0]):
    x1 = np.array(([0],X[i],Y[i]))
    x = x1.reshape(x1.shape[0],1)
    #print(x)
    k = w.T.dot(x)
    if k > 0  and abs(k)>=1:
        op1[i] = 1
    elif k < 0  and abs(k)>=1 :
        op1[i] = 0
            
#print("Done")
#print("Distance = ",dist)
print(w)
#w = w/(math.sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]))

cl1 = cl2 = pcl1 = pcl2 = 0

for i in range(dt.shape[0]):
    if op1[i] == 1:
        pcl1 = pcl1 + 1
    else:
        pcl2 = pcl2 + 1
    if op[i] == 1:
        cl1 = cl1 + 1
    else:
        cl2 = cl2 + 1
        
print(f'true class 1 : {cl1}, class 2 : {cl2}')
print(f'predicted class 1 : {pcl1}, class 2 : {pcl2}')

x1 = [min(X[:,0]), max(X[:,0])]
m = -w[1]/w[2]
c = -w[0]/w[2]
x2 = m*x1 + c
    
# Plotting
plt.scatter(X, Y,s=5,color=color)
plt.xlabel("x")
plt.ylabel("y")
plt.title('Perceptron Algorithm')
plt.legend(['Red - Class 1 & Green - Class 0'], loc=2)
plt.plot(x1, x2, 'y-')
plt.show()
