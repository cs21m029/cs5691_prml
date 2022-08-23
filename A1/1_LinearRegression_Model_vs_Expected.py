# Assignment 1.1a
# This program generates Model Vs Expected output of 

# Importing the necessary libraries
from matplotlib import pyplot as plt
#from scipy.interpolate import make_interp_spline
import numpy as np
import pandas as pd

# change 'train' to 'test' to change graph output
data = pd.read_csv("Q1_linear_reg_test_data.csv")
dt = np.array(data)

# Enter 0 to exit program
while(1):

    degree = int(input("Enter degree / Enter 0 to exit:"))

    if(degree == 0):
        break

    # Preparing X and y data from the given data
    x = dt[:, 0].reshape(dt.shape[0], 1)
    X = x
    for i in range(degree-1):
        x1 = pow(x,i+2)
        X = np.append(X, x1, axis=1)

    X = np.append(X, np.ones((dt.shape[0], 1)), axis=1)
    y = dt[:, 1].reshape(dt.shape[0], 1)

    print("Augmented X size : ",X.shape)
    print("y size",y.shape)

    # Calculating the parameters using the least square method
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    print(f'The degree {degree} parameters : {theta}')

    y_exp = np.ones((dt.shape[0], 1))
    
    lse=0.0
    #Least squared error
    for i in range(X.shape[0]):
        y_exp[i] = X[i].dot(theta)
        lse = pow(y[i]-y_exp[i],2)
       
    lse = lse/dt.shape[0]

    #plt.scatter(x,y_exp,s=5)
    plt.scatter(y_exp,y,s=5,color="red")
    plt.title('Model Output vs Expected Output - Test Data')
    plt.xlabel('Model Output')
    plt.ylabel('Expected Output')
    plt.show()
