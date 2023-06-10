#!/bin/python3
# https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow/programming-exercises
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

def build_model( learning_rate ):
    # sequential models are the simplest
    model = tf.keras.models.Sequential()
    
    # describe model topography (1 node, 1 layer)
    model.add( tf.keras.layers.Dense( units = 1, input_shape = (1,) ) )

    # compile model
    model.compile(
        optimizer = tf.keras.optimizers.experimental.RMSprop( learning_rate = learning_rate ),
        loss = "mean_squared_error",
        metrics = [tf.keras.metrics.RootMeanSquaredError()] )
    return model

def train_model( model, feature, label, epochs, batch_size ):
    # train the model
    history = model.fit( x = feature, y = label, batch_size = batch_size, epochs = epochs )

    # gather model parameters
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    epochs = history.epoch
    hist = pd.DataFrame( history.history )
    rmse = hist["root_mean_squared_error"]

    # done
    return trained_weight, trained_bias, epochs, rmse

def plot_the_model( trained_weight, trained_bias, feature, label ):
    # label axes
    plt.xlabel( "feature" )
    plt.ylabel( "label" )

    # plot features vs labels
    plt.scatter( feature, label )
    
    # create a red line representing the model
    x0 = 0
    y0 = trained_bias
    x1 = feature[-1]
    y1 = trained_bias + (trained_weight * x1 )
    plt.plot( [x0, x1], [y0, y1], c='r' )
    plt.show()

def plot_the_loss_curve(epochs, rmse):
  """Plot the loss curve, which shows loss vs. epoch."""
  plt.figure()
  plt.xlabel( "Epoch" )
  plt.ylabel( "Root Mean Squared Error" )

  plt.plot( epochs, rmse, label="Loss" )
  plt.legend()
  plt.ylim( [rmse.min()*0.97, rmse.max()] )
  plt.show()

# define hyper parameters
feature = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
label   = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])
learning_rate = .14
epochs = 70
batch_size = 1

# build & train model
model = build_model( learning_rate )
trained_weight, trained_bias, epochs, rmse = train_model( model, feature, label, epochs,  batch_size )

# plot results
plot_the_model( trained_weight, trained_bias, feature, label )
plot_the_loss_curve( epochs, rmse )
