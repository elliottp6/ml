#!/bin/python3
# GUIDE: https://www.youtube.com/watch?v=l-CjXFmcVzY
# 29:20 to ???
# limitations w/o convolution: words like 'cats' is position-dependent in original text, so it's just memorizing
# convolution makes it non-position-dependent
import numpy as np
import matplotlib.pylab as plt
import torch
from torch.nn import functional as F

# cat poem
text = '''
In cozy corners, where sunlight beams,
A creature stirs, a feline it seems.
With eyes so bright, a curious gaze,
A whiskered friend, enchanting our days.

Cats, cats, with tails that sway,
Graceful and nimble as they play.
Cats, cats, so soft and sly,
With secrets hidden in each eye.

Prowling in silence, oh, how they sneak,
Through moonlit nights, their stealth they speak.
Whiskers twitching, ears alert,
Hunting prowess, they expertly assert.

Cats, cats, with velvet paws,
They tread with elegance, no noise, no flaws.
Cats, cats, in shadows they dwell,
A mystic presence, we can't quite tell.

With gentle purrs, they claim their space,
Napping on cushions or a cherished place.
Lulled by a rhythm, soft as a breeze,
Dreaming of mice or climbing tall trees.

Cats, cats, curled in a ball,
Snuggled and snug, they never fall.
Cats, cats, their beauty is grand,
The king of the home, a regal band.

Yet, they're not just pets, they're family indeed,
Caring companions, a friend in need.
With tender mews and comforting licks,
Love in their hearts, their spirits, like flicks.

So cherish these creatures, so mystifying,
Their enchantment, forever gratifying.
Cats, cats, with love we adore,
A bond unbroken, forevermore.
'''

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
