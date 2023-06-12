#!/bin/python3
#https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/intro_to_neural_nets.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=intro_to_nn_tf2-colab&hl=en
import numpy as p
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns

# adjust granularity of reporting
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# gather data
train_df = pd.read_csv( "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv" )
train_df = train_df.reindex( np.random.permutation( train_df.index ) ) # shuffle the examples
test_df = pd.read_csv( "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv" )

# 
