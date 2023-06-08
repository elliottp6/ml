#!/bin/python3
# GUIDE: https://www.youtube.com/watch?v=l-CjXFmcVzY
# 23:50 to 29:20
import numpy as np
import matplotlib.pylab as plt
import torch
from torch.nn import functional as F

# read cat poem
with open( 'cat_poem.txt' ) as h: text = h.read().lower()

# get characters
chars = sorted( list( set( text ) ) )

# build dictionary which converts character to integer, and vice-versa
stoi = {ch:i for i,ch in enumerate( chars )}
itos = {i:ch for i,ch in enumerate( chars )}

# convert text to integers (data)
data = [stoi[c] for c in text]

# convert data to tensor
data = torch.tensor( data ).float()

# print vocabulary size
vocab_size = len( stoi )
print( "vocabulary size: " + str( vocab_size ) )

# define constants
inputs = 64
hidden1 = 200
hidden2 = 200
outputs = vocab_size

# add bias to input??
#xs = np.hstack( (xs, np.ones( [xs.shape[0], 1] ) ) )
#inputs += 1

# define initial weights as tensors
params = []
def weights( ins, outs ):
    ws = torch.randn( ins, outs ) * 0.1 # the * .1 is to regularize, which means start w/ small weights so random luck plays less of a role (works really well with relu)
    ws = ws.requires_grad_( True )
    params.append( ws )
    return ws

class Model():
    def __init__( self):
        self.w0 = weights( inputs, hidden1 )
        self.w1 = weights( hidden1, hidden2 )
        self.w2 = weights( hidden2, outputs )

    def forward( self, x ):
        x = torch.relu( x @ self.w0 )
        x = torch.relu( x @ self.w1 )
        return x @ self.w2

# create the model & the optimizer
model = Model()
optimizer = torch.optim.Adam( params, 0.001 )

# main training loop
errors = []
for i in range( 5000 ):
    # sample the data
    b = torch.randint( len( data ) - inputs, (100,) )
    xs = torch.stack( [data[i:i+inputs] for i in b] )
    ys = torch.stack( [data[i+inputs:i+inputs+1] for i in b] )
    
    # forward pass
    yh = model.forward( xs )

    # backward pass
    #loss = F.mse_loss( yh, ys )
    loss = F.cross_entropy( yh.view( -1, vocab_size ), ys.long().view( -1 ) )
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
#plt.yscale( "log" )
plt.plot( errors )

# define function plot
with torch.no_grad(): # https://stackoverflow.com/questions/55466298/pytorch-cant-call-numpy-on-variable-that-requires-grad-use-var-detach-num
    plt.figure( 2 )
    plt.plot( ys )
    yh = torch.argmax( yh, dim=-1 ) # select max probabilty output as our prediction
    plt.plot( yh )

# show plots
plt.show()

# generate text, starting w/ first block of poem
gen_text = text[0:inputs]
s = data[0:inputs]
for i in range( 3000 ):
    # get vector representing likeliness for each class (in this case, letters)
    yh = model.forward( s )
    
    # DISABLED: convert vector into the highest likeliness letter
    #pred = torch.argmax( yh ).item()
    
    # ENABLED: convert vector into a letter based on a random sampling of output probabilities
    prob = F.softmax( yh, dim=0 ) # softmax converts numbers/logits into a probability vector
    pred = torch.multinomial( prob, num_samples=1 ).item() # this does the random sampling from the probability vector

    # insert predicted letter into input (like a ring buffer)
    s = torch.roll( s, -1 )
    s[-1] = pred

    # convert 'pred' to letter
    gen_text += itos[pred]
    
# print it out
print( gen_text )
