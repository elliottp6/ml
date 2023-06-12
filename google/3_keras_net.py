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
# define plotting function for loss curve
def plot_the_loss_curve( epochs, mse_training, mse_validation ):
    """Plot a curve of loss vs. epoch."""
    plt.figure()
    plt.xlabel( "Epoch" )
    plt.ylabel( "Mean Squared Error" )

    plt.plot( epochs, mse_training, label="Training Loss" )
    plt.plot( epochs, mse_validation, label="Validation Loss" )

    # mse_training is a pandas Series, so convert it to a list first.
    merged_mse_lists = mse_training.tolist() + mse_validation
    highest_loss = max( merged_mse_lists )
    lowest_loss = min( merged_mse_lists )
    top_of_y_axis = highest_loss * 1.03
    bottom_of_y_axis = lowest_loss * 0.97 

    plt.ylim( [bottom_of_y_axis, top_of_y_axis] )
    plt.legend()
    plt.show()  

# simple linear models are a good baseline before creating a deep neural net
def create_linear_model( inputs, outputs, learning_rate ):
    # create the model
    model = tf.keras.Model( inputs = inputs, outputs = outputs )
    
    # compile model with adam optimizer
    model.compile( optimizer = tf.keras.optimizers.Adam(
        learning_rate = learning_rate),
        loss = "mean_squared_error",
        metrics = [tf.keras.metrics.MeanSquaredError()] )
    return model

def train_linear_model( model, dataset, epochs, batch_size, label_name, validation_split = 0.1 ):
    """Feed a dataset into the model in order to train it."""

    # Split the dataset into features and label.
    features = {name:np.array(value) for name, value in dataset.items()}
    label = train_median_house_value_normalized( np.array( features.pop( label_name ) ) )
    history = model.fit( x = features, y = label, batch_size = batch_size,
                         epochs = epochs, shuffle = True, validation_split = validation_split)

    # Get details that will be useful for plotting the loss curve.
    epochs = history.epoch
    hist = pd.DataFrame( history.history )
    mse = hist["mean_squared_error"]
    return epochs, mse, history.history

#@title Define linear regression model outputs
def get_outputs_linear_regression():
    # Create the Dense output layer.
    dense_output = tf.keras.layers.Dense( units=1, input_shape=(1,), name='dense_output' )(preprocessing_layers)

    # Define an output dictionary we'll send to the model constructor.
    outputs = { 'dense_output': dense_output }
    return outputs

# normalize the house values
train_median_house_value_normalized = tf.keras.layers.Normalization( axis = None )
test_median_house_value_normalized = tf.keras.layers.Normalization( axis = None )
train_median_house_value_normalized.adapt( np.array( train_df['median_house_value'] ) )
test_median_house_value_normalized.adapt( np.array( test_df['median_house_value'] ) )

# The following variables are the hyperparameters.
learning_rate = 0.01
epochs = 15
batch_size = 1000
label_name = "median_house_value"

# Split the original training set into a reduced training set and a validation set. 
validation_split = 0.2
outputs = get_outputs_linear_regression()

# Establish the model's topography.
my_model = create_linear_model( inputs, outputs, learning_rate )

# Train the model on the normalized training set.
epochs, mse, history = train_linear_model( my_model, train_df, epochs, batch_size, label_name, validation_split )
plot_the_loss_curve( epochs, mse, history["val_mean_squared_error"] )

test_features = {name:np.array(value) for name, value in test_df.items()}
test_label = test_median_house_value_normalized( test_features.pop(label_name) ) # isolate the label
print( "\n Evaluate the linear regression model against the test set:" )
my_model.evaluate( x = test_features, y = test_label, batch_size=batch_size, return_dict = True )

