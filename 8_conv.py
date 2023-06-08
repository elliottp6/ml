#!/bin/python3
# GUIDE: https://www.youtube.com/watch?v=l-CjXFmcVzY
# 29:20 to ???
# limitations w/o convolution: words like 'cats' is position-dependent in original text, so it's just memorizing
# convolution makes the network invariant to position, rotation & scaling
# embedding: each character has a unique vector INSTEAD of just a single value
#            then pass embedding into a filter: a simple linear network
#            then move the filter, and do the same for the next letter in the context
#            sum all the outputs and pass result into a nonlinear network => output is next letter
#            network can learn patterns regardless of their position
import numpy as np
import matplotlib.pylab as plt
import torch
from torch.nn import functional as F

# read cat poem
with open( 'cat_poem.txt' ) as h: text = h.read().lower()

# change to lowercase to reduce the # of characters
text = text.lower()

# get characters
chars = sorted( list( set( text ) ) )

# build dictionary which converts character to integer, and vice-versa
stoi = {ch:i for i,ch in enumerate( chars )}
itos = {i:ch for i,ch in enumerate( chars )}

# convert text to integers (data)
data = [stoi[c] for c in text]

# convert data to tensor
data = torch.tensor( data ).long() # change to LONG b/c now using data as indices for embedding table

# print vocabulary size
vocab_size = len( stoi )
print( "vocabulary size: " + str( vocab_size ) )

# define constants
inputs = 64
hidden1 = 200
hidden2 = 200
outputs = vocab_size
lr = 0.001
num_embed = 64
embed = torch.randn( vocab_size, num_embed ) # associate each vocabulary w/ a random vector


# define initial weights as tensors
params = []
def weights( ins, outs ):
    ws = torch.randn( ins, outs ) * 0.1 # the * .1 is to regularize, which means start w/ small weights so random luck plays less of a role (works really well with relu)
    ws = ws.requires_grad_( True )
    params.append( ws )
    return ws

class Model():
    def __init__( self):
        self.wv = weights( num_embed, num_embed )
        self.w0 = weights( num_embed, hidden1 )
        self.w1 = weights( hidden1, hidden2 )
        self.w2 = weights( hidden2, outputs )

    def forward( self, x ):
        x = embed[x] # lookup embedding vector for character
        x = x @ self.wv # linear layer
        x = torch.sum( x, dim=-2 ) # sum 
        x = torch.relu( x @ self.w0 ) # nonlinear layer
        x = torch.relu( x @ self.w1 ) # nonlinear layer
        return x @ self.w2

# create the model & the optimizer
model = Model()
optimizer = torch.optim.Adam( params, lr )

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
    # get next predicted letter
    yh = model.forward( s )
    prob = F.softmax( yh, dim=0 )
    #pred = torch.argmax( yh ).item()
    pred = torch.multinomial( prob, num_samples=1 ).item()

    # insert predicted letter into input (like a ring buffer)
    s = torch.roll( s, -1 )
    s[-1] = pred

    # convert 'pred' to letter
    gen_text += itos[pred]
    
# print it out
print( gen_text )
