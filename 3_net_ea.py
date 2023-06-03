#!/bin/python3
# GUIDE: https://www.youtube.com/watch?v=l-CjXFmcVzY
# 6:22 to 9:00
# perceptron, trained by evolving weights
import numpy as np
import matplotlib.pyplot as plt

# deterministic runs are easier
np.random.seed( 0 )

# create weights
def weights( ins, outs ): return np.random.randn( ins, outs )

# constants
num_inputs = 6
num_hidden = 5
num_outputs = 1

# inputs (note that last value is just a bias)
# it could have been later inserted like this:
#xs = np.hstack( (xs, np.ones( [xs.shape[0], 1] ) ) )
#num_inputs += 1
xs = np.asarray( [[0, 1, 0, 1, 0, 1],
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
wi = weights( num_inputs, num_hidden )
ws = weights( num_hidden, num_outputs )

# main training loop
errors = []
min_errors = []
min_error = float( 'inf' )
for i in range( 5000 ):
    # apply mutation
    wi_mutated = wi + weights( num_inputs, num_hidden ) * 0.02
    ws_mutated = ws + weights( num_hidden, num_outputs ) * 0.02
    x = xs @ wi_mutated # apply initial weights
    x = np.sin( x ) # sine wave nonlinear transform
    #x = np.maximum( 0, x ) # relu nonlinear transform
    yh = x @ ws_mutated # hidden layer
    e = yh - ys
    e = np.sum( np.abs( e ) )
    
    # track errors
    errors.append( e )
    min_errors.append( min_error )

    # check if we have a better solution
    if e < min_error:
        min_error = e
        ws = ws_mutated
        wi = wi_mutated
        if e < 0.05:
            print( "found solution" )
            print( "wi: " + str( wi ) )
            print( "ws: " + str( ws ) )
            break

# show plot of errors
print( "min error: " + str( min_error ) )
plt.figure( 1 )
plt.plot( errors )
plt.plot( min_errors )
plt.show()
