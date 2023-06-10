#!/bin/python3
# https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow/programming-exercises
import numpy as np
import pandas as pd

# --NUMPY ULTRAQUICK TUTORIAL -- https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/numpy_ultraquick_tutorial.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=numpy_tf2-colab&hl=en
# cheat sheet
one_dimensional_array = np.array( [1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5] )
two_dimensional_array = np.array( [[6, 5], [11, 7], [4, 8]] )
ones = np.ones( 3 )
zeros = np.zeros( (3,3) )
sequence_of_integers = np.arange( 5, 12 ) # includes lower bound but not upper bound
random_integers_between_50_and_100 = np.random.randint( low=50, high=101, size=(6) )
random_floats_between_0_and_1 = np.random.random( [6] )
random_floats_between_2_and_3 = random_floats_between_0_and_1 + 2.0
random_integers_between_150_and_300 = random_integers_between_50_and_100 * 3

# Task 1
feature = np.arange( 6, 21 )
label = (feature * 3) + 4

# Task 2
noise = np.random.random( [len(feature)] )
label = label + noise * 4 - 2 # note that '+=' cannot change the dtype (datatype) of an array, and we're converting to a float array here

# -- PANDAS ULTRAQUICK TUTORIAL -- https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/pandas_dataframe_ultraquick_tutorial.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=pandas_tf2-colab&hl=en#scrollTo=ZmL0l551Iibq
data = np.array( [[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]] ) # 5x2 numpy array
columns = ['temperature', 'activity']
df = pd.DataFrame( data = data, columns = columns )

# add column 'adjusted
df['adjusted'] = df['activity'] + 2

print( df )
##   temperature  activity  adjusted
##0            0         3         5
##1           10         7         9
##2           20         9        11
##3           30        14        16
##4           40        15        17

# print subsets of dataframe
print( "Rows #0, #1, and #2:" )
print( df.head( 3 ), '\n' )

print( "Row #2:" )
print( df.iloc[[2]], '\n' )

print( "Rows #1, #2, and #3:" )
print( df[1:4], '\n' )

print( "Column 'temperature':" )
print( df['temperature'] )

# reference temperature at 0
print( df['temperature'][0] )

# Create a true copy of my_dataframe
copy = df.copy()

# Modify a cell in df, but note that copy does not change
df.at[1, 'temperature'] = 555
print( df )
print( copy )
