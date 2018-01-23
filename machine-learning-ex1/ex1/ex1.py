"""Same code for linear regression in python,
Exercise 1 Linear regression"""


#Initialize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
from ex1_functions import *

#Creating Identity matrix
print("5*5 identity matrix\n")
print (warmUpExercise())

#Plotting data
print("plotting data \n")
data = pd.read_csv("ex1data1.txt",names=["X","y"])
x = np.array(data.X)[:,None]
y = np.array(data.y)
m = len(y)
fig = plotData(x,y)
fig.show()

#Gradient descent
X = np.c_[np.ones(m), x]	#Add a column of ones's before x, similarly if want to add after x, do [x, np.ones(m)]
theta = np.zeros(2)	#Initialize parameters

iterations = 1500
alpha = 0.01

computeCost(X, y, theta)

#Run gradien descent
theta, hist = gradientDescent(X, y, theta, alpha, iterations)

print('Theta found by gradient descent: ')
print(theta[0],"\n", theta[1])

# Plot the linear fit
plt.plot(x,y,'rx',x,np.dot(X,theta),'b-')
plt.legend(['Training Data','Linear Regression'])
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5],theta) # takes inner product to get y_bar
print('For population = 35,000, we predict a profit of ', predict1*10000)

predict2 = np.dot([1, 7],theta)
print('For population = 70,000, we predict a profit of ', predict2*10000)








"""
#<-------------------------------------------------------------------Still need to learn----------------------------->

#Visualizing J
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J 
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))

# Fill out J_Vals 
# Note: There is probably a more efficient way to do this that uses
#	broadcasting instead of the nested for loops
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i],theta1_vals[j]])
        J_vals[i][j] = computeCost(X,y,t)


# Surface plot using J_Vals
fig = plt.figure()
ax = plt.subplot(111,projection='3d')
Axes3D.plot_surface(ax,theta0_vals,theta1_vals,J_vals,cmap=cm.coolwarm)
plt.show()

# Contour plot
# TO DO: Currently does not work as expected. Need to find a way to mimic
#	 the logspace option in matlab
fig = plt.figure()
ax = plt.subplot(111)
plt.contour(theta0_vals,theta1_vals,J_vals) """