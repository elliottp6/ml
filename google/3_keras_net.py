#!/bin/python3
#https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/intro_to_neural_nets.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=intro_to_nn_tf2-colab&hl=en
# NOTE: after this, then start on multiclass net example: https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/multi-class_classification_with_MNIST.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=multiclass_tf2-colab&hl=en
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns

# adjust granularity of reporting
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# -- STEP 1: GRAB DATA AND CREATE INPUT TENSORS --

# gather data
train_df = pd.read_csv( "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv" )
train_df = train_df.reindex( np.random.permutation( train_df.index ) ) # shuffle the examples
test_df = pd.read_csv( "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv" )

# create input tensors
inputs = { 
    'latitude': tf.keras.layers.Input( shape=(1,), dtype=tf.float32, name='latitude' ),
    'longitude': tf.keras.layers.Input( shape=(1,), dtype=tf.float32, name='longitude'),
    'median_income': tf.keras.layers.Input( shape=(1,), dtype=tf.float32, name='median_income'),
    'population': tf.keras.layers.Input( shape=(1,), dtype=tf.float32, name='population')
}

print( 'Data fetched and input tensors created' )

# -- STEP 2: CREATE-PREPROCESSING LAYERS --

# create normalization layers
median_income = tf.keras.layers.Normalization( name = 'normalization_median_income', axis = None )
population = tf.keras.layers.Normalization( name = 'normalization_population', axis = None )
latitude = tf.keras.layers.Normalization( name='normalization_latitude', axis = None )
longitude = tf.keras.layers.Normalization( name='normalization_longitude', axis = None )

# adapt layers to data (i.e. recenter & scale s.t. it is between [0,1] or [-1,1], potentially subtract mean and divide by standard deviation, to convert to standard normal distribution where values are now Z-scores from appox -3 to +3
median_income.adapt( train_df['median_income'] )
population.adapt( train_df['population'] )
latitude.adapt( train_df['latitude'] )
longitude.adapt( train_df['longitude'] )

# ???
median_income = median_income( inputs.get('median_income') )
population = population( inputs.get( 'population' ) )
latitude = latitude( inputs.get('latitude' ) )
longitude = longitude( inputs.get('longitude') )

# define buckets for latitude & longitude: in this case, for the Z-scores (-3 to +3), and we can 20 buckets (i.e. 21 edges)
latitude_boundaries = np.linspace( -3, 3, 21 ) # linspace returns evenly spaced numbers over the specified internal
longitude_boundaries = np.linspace( -3, 3, 21 )

# create discretization layers
latitude = tf.keras.layers.Discretization( bin_boundaries = latitude_boundaries, name = 'discretization_latitude')(latitude)
longitude = tf.keras.layers.Discretization( bin_boundaries = longitude_boundaries, name = 'discretization_longitude')(longitude)

# cross latitude & longitude features into a single one-hot vector
feature_cross = tf.keras.layers.HashedCrossing(
    num_bins = len( latitude_boundaries ) * len( longitude_boundaries ), 
    output_mode = 'one_hot',
    name = 'cross_latitude_longitude')([latitude, longitude])

# concatenate inputs into a single tensor
preprocessing_layers = tf.keras.layers.Concatenate()( [feature_cross, median_income, population] )

print( "Preprocessing layers defined." )

# -- STEP 3: ... --
