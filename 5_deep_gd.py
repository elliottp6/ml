#!/bin/python3
# GUIDE: https://www.youtube.com/watch?v=l-CjXFmcVzY
# 12:12 to 15:30
# perceptron, trained by gradient descent
import numpy as np
import matplotlib.pyplot as plt

# define inputs and outputs for non-linear function x^2
x0 = np.asarray( [[-10], [-8], [-6], [-4], [-2], [0], [2], [4], [6], [8], [10]] )
y = x0**2

# define constants
num_inputs = 1
num_hidden1 = 20
num_hidden2 = 20
num_outputs = 1
lr = 0.000005 # learning rate

# add bias to input
x0 = np.hstack( (x0, np.ones( [x0.shape[0], 1] ) ) )
num_inputs += 1

# define initial weights
def weights( ins, outs ): return np.random.randn( ins, outs )
np.random.seed( 0 )
w0 = weights( num_inputs, num_hidden1 )
w1 = weights( num_hidden1, num_hidden2 )
w2 = weights( num_hidden2, num_outputs )

# main training loop
errors = []
for i in range( 50000 ):
    # forward pass
    z0 = x0 @ w0;   x1 = np.sin( z0 )
    z1 = x1 @ w1;   x2 = np.sin( z1 )
    z2 = x2 @ w2

    # backward pass
    e2 = z2 - y
    e1 = (e2 @ w2.T) * np.cos( z1 )
    e0 = (e1 @ w1.T) * np.cos( z0 )

    # update weights
    w2 -= (x2.T @ e2) * lr
    w1 -= (x1.T @ e1) * lr
    w0 -= (x0.T @ e0) * lr  
    
    # track error
    errors.append( np.sum( np.abs( e2 ) ) )

# define error plot
print( "min error: " + str( errors[-1] ) )
plt.figure( 1 )
plt.yscale( "log" )
plt.plot( errors )

# define 
plt.figure( 2 )
plt.plot( y )
plt.plot( z2 )

# show plots
plt.show()
