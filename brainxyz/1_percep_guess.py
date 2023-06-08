#!/bin/python3
# GUIDE: https://www.youtube.com/watch?v=l-CjXFmcVzY
# 0:00 to 5:12
# perceptron, trained by guessing weights
import numpy as np
import matplotlib.pyplot as plt

# inputs
ins = 5
xs = np.asarray( [[0, 1, 0, 1, 0],
                  [0, 0, 1, 1, 0],
                  [1, 1, 0, 1, 0],
                  [1, 1, 1, 0, 1],
                  [0, 0, 0, 1, 0]] )

# outputs
outs = 1
ys = np.asarray( [[0],
                  [1],
                  [1],
                  [1],
                  [0]] )

# create weights
def weights( ins, outs ):
    ws = np.random.randn( ins, outs )
    return ws

# initial guessed weights
ws = weights( ins, outs )

# compute errors using weights
errors = []
for i in range( 5000 ):
    # compute output
    yh = xs @ ws

    # record error
    e = yh - ys
    e = np.sum( np.abs( e ) )
    errors.append( e )

    # if error is low enough, we found a solution
    if e < 0.05:
        print( "found solution" )
        print( ws )
        break

    # otherwise, guess new weights
    ws = weights( ins, outs )

# show plot of errors
plt.figure( 1 )
plt.plot( errors )
plt.show()
