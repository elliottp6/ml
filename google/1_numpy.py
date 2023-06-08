#!/bin/python3
# https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow/programming-exercises
# https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/numpy_ultraquick_tutorial.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=numpy_tf2-colab&hl=en
import numpy as np

one_dimensional_array = np.array( [1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5] )
two_dimensional_array = np.array( [[6, 5], [11, 7], [4, 8]] )
sequence_of_integers = np.arrange( 5, 12 ) # includes lower bound but not upper bound
random_integers_between_50_and_100 = np.random.randint( low=50, high=101, size=(6) )
random_floats_between_0_and_1 = np.random.random( [6] )
