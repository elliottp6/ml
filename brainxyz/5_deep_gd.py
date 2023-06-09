#!/bin/python3
# GUIDE: https://www.youtube.com/watch?v=l-CjXFmcVzY
# 12:12 to 17:15
# perceptron, trained by gradient descent
# issues to note: vanishing or exploding graident problems (16:09) where blame is too much on initial weights or final weights
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
    z0 = x0 @ w0;   x1 = np.sin( z0 )   # sine transform, derivative = (∂x1/∂z0) = cos(z0)
    z1 = x1 @ w1;   x2 = np.sin( z1 )   # sine transform, derivative = (∂x2/∂z1) = cos(z1)
    z2 = x2 @ w2;   x3 = z2             # identity transform, derivative = (∂x3/∂z2) = 1

    # backward pass via chain rule: f(g(x))' = f'(g(x))g'(x)
    C = x3 - y                      # C, cost function, note that derivative of cost (∂C/∂x3) = 1
    e2 = C * 1 * 1                  # e2, units (∂C/∂z2) = C        * (∂C/∂x3)  * (∂x3/∂z2) = C * 1 * 1
    e1 = (e2 @ w2.T) * np.cos( z1 ) # e1, units (∂C/∂z1) = (∂C/∂z2) * (∂z2/∂x2) * (∂x2/∂z1) = e2 * w2 * cos(z1)
    e0 = (e1 @ w1.T) * np.cos( z0 ) # e0, units (∂C/∂z0) = (∂C/∂z1) * (∂z1/∂x1) * (∂x1/∂z0) = e1 * w1 * cos(z0)

    # update weights
    w2 -= (x2.T @ e2) * lr          # w2, units (∂C/∂w2) = (∂C/∂z2) * (∂z2/∂w2) = e2 * x2
    w1 -= (x1.T @ e1) * lr          # w1, units (∂C/∂w1) = (∂C/∂z1) * (∂z1/∂w1) = e1 * x1
    w0 -= (x0.T @ e0) * lr          # w0, units (∂C/∂w0) = (∂C/∂z0) * (∂z0/∂w0) = e0 * x0
    
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
