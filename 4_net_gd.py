#!/bin/python3
# GUIDE: https://www.youtube.com/watch?v=l-CjXFmcVzY
# 9:00 to 12:12
# perceptron, trained by gradient descent
import numpy as np
import matplotlib.pyplot as plt

# deterministic runs are easier
np.random.seed( 0 )

# create weights
def weights( ins, outs ): return np.random.randn( ins, outs )

# constants
num_inputs = 6
num_hidden = 1
num_outputs = 1
learning_rate = 0.01

# inputs (note that last value is just a bias)
# it could have been later inserted like this:
#xs = np.hstack( (xs, np.ones( [xs.shape[0], 1] ) ) )
#num_inputs += 1
x0 = np.asarray( [[0, 1, 0, 1, 0, 1],
                  [0, 0, 1, 1, 0, 1],
                  [1, 1, 0, 1, 0, 1],
                  [1, 1, 1, 0, 1, 1],
                  [0, 0, 0, 1, 0, 1]] )

# outputs
ys = np.asarray( [[0],
                  [0],
                  [0],
                  [3],
                  [3]] )

# define weights
w0 = weights( num_inputs, num_hidden )
w1 = weights( num_hidden, num_outputs )

# main training loop
errors = []
for i in range( 5000 ):
    # forward pass
    z0 = x0 @ w0;   x1 = np.sin( z0 )
    yh = x1 @ w1

    # backward pass
    e1 = yh - ys
    e0 = (e1 @ w1.T) * np.cos( z0 )

    # update weights
    w1 -= (x1.T @ e1) * learning_rate
    w0 -= (x0.T @ e0) * learning_rate  
    
    # track error
    errors.append( np.sum( np.abs( e1 ) ) )

# show plot of errors
print( "min error: " + str( errors[-1] ) )
plt.figure( 1 )
plt.yscale( "log" )
plt.plot( errors )
plt.show()
