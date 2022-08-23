# Assignment 1.1a

# Importing the necessary libraries
from matplotlib import pyplot as plt
#from scipy.interpolate import make_interp_spline
import numpy as np
import pandas as pd

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

    print(X.shape)
    print(y.shape)

    #we got parameter from training program
    #for degree 5
    theta1 = np.array(([ 8.45327805e+00],[-1.61482012e+01],[ 2.19295090e+01],[-1.66255571e+01],[ 7.57060240e+00],[-2.19650555e+00], [ 4.15148616e-01],[-5.08344391e-02],[ 3.88427159e-03],[-1.68211137e-04],[ 3.15100662e-06],[-3.92315045e-01]))

    #for degree 11
    #theta1 = np.array(([ 2.78941392e+00],[ 5.33713746e-01],[-4.92997130e-01],[ 6.86379156e-02],[-2.75075998e-03],[ 1.21867224e-01]))

    
    theta = theta1.reshape(theta1.shape[0],1)

    print(f'The parameters : {theta}')

    y_exp = np.ones((dt.shape[0], 1))
    
    lse=0.0
    #Least squared error
    for i in range(X.shape[0]):
        y_exp[i] = X[i].dot(theta)
        lse = pow(y[i]-y_exp[i],2)
       
    lse = lse/dt.shape[0]

    print(f'Least Squared Error at degree {degree} : {lse}')

    # Now, calculating the y-axis values against x-values according to
    # the parameters theta0 and theta1
    y_line = X.dot(theta)

    # Plotting the data points and the best fit line
    plt.scatter(x, y,s=5,c='red')

    xs = np.linspace(0,10,500)
    xs1 = xs[:].reshape(xs.shape[0], 1)
    Xs=xs1
    for i in range(degree-1):
        x1 = pow(xs1,i+2)
        Xs = np.append(Xs, x1, axis=1)
        
    Xs = np.append(Xs, np.ones((xs1.shape[0], 1)), axis=1)
    print(Xs.shape)
    ys= Xs.dot(theta)
    plt.plot(xs1,ys)
    plt.title('Linear Regression Best Fit curve - Test Data')
    plt.xlabel('Feature x')
    plt.ylabel('Feature y')
    plt.show()
