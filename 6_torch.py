#!/bin/python3
# instlal pytorch via https://pytorch.org/get-started/locally/
# GUIDE: https://www.youtube.com/watch?v=l-CjXFmcVzY
# 17:15 to 23:50
# conclusions: nets can fir & interpolate well, but not extrapolate well
# note: to verify pytorch is working do:
#x = torch.rand( 5, 3 )
#print( x )
import numpy as np
import matplotlib.pylab as plt
import torch
from torch.nn import functional as F

# define constants
num_inputs = 1
num_hidden1 = 20
num_hidden2 = 20
num_outputs = 1

# define inputs and outputs for non-linear function x^2
xs = np.asarray( [[-10], [-8], [-6], [-4], [-2], [0], [2], [4], [6], [8], [10]] )
ys = xs**2

# add bias to input
xs = np.hstack( (xs, np.ones( [xs.shape[0], 1] ) ) )
num_inputs += 1

# convert numpy arrays to tensors
xs = torch.tensor( xs ).float()
ys = torch.tensor( ys ).float()

# define initial weights as tensors
params = []
def weights( ins, outs ):
    ws = torch.randn( ins, outs ) * 0.1 # the * .1 is to regularize, which means start w/ small weights so random luck plays less of a role (works really well with relu)
    ws = ws.requires_grad_( True )
    params.append( ws )
    return ws

class Model():
    def __init__( self):
        self.w0 = weights( num_inputs, num_hidden1 )
        self.w1 = weights( num_hidden1, num_hidden2 )
        self.w2 = weights( num_hidden2, num_outputs )

    def forward( self, x ):
        x = torch.relu( x @ self.w0 )
        x = torch.relu( x @ self.w1 )
        return x @ self.w2

# create the model & the optimizer
model = Model()
optimizer = torch.optim.Adam( params, 0.001 )

# main training loop
errors = []
for i in range( 10000 ):
    # forward pass
    yh = model.forward( xs )

    # backward pass
    loss = F.mse_loss( yh, ys )
    optimizer.zero_grad()
    loss.backward()

    # update weights
    optimizer.step() 
    
    # track error
    errors.append( loss.item() )
    if 0 == i % 500: print( errors[-1] )

# define error plot
print( "min error: " + str( errors[-1] ) )
plt.figure( 1 )
plt.yscale( "log" )
plt.plot( errors )

# define function plot
with torch.no_grad(): # https://stackoverflow.com/questions/55466298/pytorch-cant-call-numpy-on-variable-that-requires-grad-use-var-detach-num
    plt.figure( 2 )
    plt.plot( ys )
    plt.plot( yh )

# show plots
plt.show()

# test on unseed data
value = -5
value = torch.tensor( [value, 1] ).float()
result = model.forward( value )
print( result )
