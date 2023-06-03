#!/bin/python3
# GUIDE: https://www.youtube.com/watch?v=l-CjXFmcVzY
# 5:12 to 6:22
# perceptron, trained by evolving weights
import numpy as np
import matplotlib.pyplot as plt

# inputs
ins = 5
xs = np.asarray( [[0, 1, 0, 1, 0],
                  [0, 0, 1, 1, 0],
                  [1, 1, 0, 1, 0],
                  [1, 1, 1, 0, 1],
                  [0, 0, 0, 1, 0]] )

# weights
# ws = np.asarray( [1, 0, 1, 0, -1)] )

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
min_errors = []
min_error = float( 'inf' )
for i in range( 5000 ):
    # compute output w/ mutated weights
    ws_mutated = ws + weights( ins, outs ) * 0.02
    yh = xs @ ws_mutated
    e = yh - ys
    e = np.sum( np.abs( e ) )

    # track errors
    errors.append( e )
    min_errors.append( min_error )

    # check if we have a better solution
    if e < min_error:
        min_error = e
        ws = ws_mutated
        if e < 0.05:
            print( "found solution" )
            print( ws )
            break

# show plot of errors
print( "min error: " + str( min_error ) )
plt.figure( 1 )
plt.plot( errors )
plt.plot( min_errors )
plt.show()
