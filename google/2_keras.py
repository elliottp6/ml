#!/bin/python3
# https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow/programming-exercises
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# adjust granularity of pandas reporting
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

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

def train_model( model, df, feature, label, epochs, batch_size ):
    # train the model
    history = model.fit( x = df[feature], y = df[label], batch_size = batch_size, epochs = epochs )

    # gather model parameters
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    epochs = history.epoch
    hist = pd.DataFrame( history.history )
    rmse = hist["root_mean_squared_error"]

    # done
    return trained_weight, trained_bias, epochs, rmse

def plot_the_model( trained_weight, trained_bias, feature, label ):
    """Plot the trained model against 200 random training examples."""

    # Label the axes.
    plt.xlabel( feature )
    plt.ylabel( label )

    # Create a scatter plot from 200 random points of the dataset.
    random_examples = training_df.sample( n = 200 )
    plt.scatter( random_examples[feature], random_examples[label] )

    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    x0 = 0
    y0 = trained_bias
    x1 = random_examples[feature].max()
    y1 = trained_bias + (trained_weight * x1)
    plt.plot( [x0, x1], [y0, y1], c='r' )

    # Render the scatter plot and the red line.
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

def predict_house_values(n, feature, label):
    """Predict house values based on a feature."""

    batch = training_df[feature][10000:10000 + n]
    predicted_values = model.predict_on_batch( x = batch )

    print( "feature   label          predicted" )
    print( "  value   value          value" )
    print( "          in thousand$   in thousand$" )
    print( "--------------------------------------" )
    for i in range( n ):
        print ("%5.0f %6.0f %15.0f" % (training_df[feature][10000 + i],
                                       training_df[label][10000 + i],
                                       predicted_values[i][0] ))

# -- 2nd exercise --
# import dataset of california median house values
training_df = pd.read_csv( filepath_or_buffer = "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv" )

# scale 'median_house_val' into thousands
training_df["median_house_value"] /= 1000.0

# print data
## print( training_df.head() )
## print( training_df.describe() )

# build & train model
feature = "median_income"
label = "median_house_value"

# generate a correlation matrix, which will tell you what feature might correspond with what label
# corr = training_df.corr()
# print( corr )

model = build_model( learning_rate = 0.06 )
weight, bias, epochs, rmse = train_model( model, training_df,
                                          feature = feature,
                                          label = label,
                                          epochs = 24,
                                          batch_size = 30 )

print( "\nThe learned weight for your model is %.4f" % weight )
print( "The learned bias for your model is %.4f\n" % bias )

plot_the_model( weight, bias, feature, label )
plot_the_loss_curve( epochs, rmse )

predict_house_values( 10, feature, label )

