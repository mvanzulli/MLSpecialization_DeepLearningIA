# Import libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


# **Tensorflow and Keras**  
# Tensorflow is a machine learning package developed by Google. 
# In 2019, Google integrated Keras into Tensorflow and released Tensorflow 2.0.
#  Keras is a framework developed independently by François Chollet that creates a simple,
#  layer-centric interface to Tensorflow. This course will be using the Keras interface.


### 2.1 Problem Statement

# In this exercise, you will use a neural network to recognize two handwritten digits, zero and one. 
# This is a binary classification task. Automated handwritten digit recognition is widely used today - 
# from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks. 
# You will extend this network to recognize all 10 digits (0-9) in a future assignment. 

# This exercise will show you how the methods you have learned can be used for this classification task.

# ### 2.2 Dataset

# You will start by loading the dataset for this task. 
# - The `load_data()` function shown below loads the data into variables `X` and `y`


# - The data set contains 1000 training examples of handwritten digits $^1$, here limited to zero and one.  

#     - Each training example is a 20-pixel x 20-pixel grayscale image of the digit. 
#         - Each pixel is represented by a floating-point number indicating the grayscale intensity at that location. 
#         - The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector. 
#         - Each training example becomes a single row in our data matrix `X`. 
#         - This gives us a 1000 x 400 matrix `X` where every row is a training example of a handwritten digit image.



# Sequential model
model = Sequential(
    [               
        tf.keras.Input(shape=(400,)),  
        tf.keras.layers.Dense(25, activation = 'ReLu', name="layer1"),
        tf.keras.layers.Dense(15, activation = 'ReLu', name="layer2"),
        tf.keras.layers.Dense(1, activation = 'sigmoid', name="layer3"),
        
    ], name = "my_model" 
)        

# Exercise 2

# Below, build a dense layer subroutine. The example in lecture utilized a for loop
#  to visit each unit (`j`) in the layer and perform the dot product of the weights for that unit (`W[:,j]`) 
# and sum the bias for the unit (`b[j]`) to form `z`. An activation function `g(z)` is then applied to that result.
#  This section will not utilize some of the matrix operations described in the optional lectures. These will be explored in a later section.


def my_dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units
    """
    units = W.shape[1]
    a_out = np.zeros(units)

    for j in range(units): 
        y = np.dot(W[:,j],a_in) + b[j]
        print(y)
        a_out[j] = g(y)
    
    return(a_out)

# Exercise 3


# The following cell builds a three-layer neural network utilizing the 
# `my_dense` subroutine above.

def my_sequential(x, W1, b1, W2, b2, W3, b3):
    a1 = my_dense(x,  W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)
    return(a3)