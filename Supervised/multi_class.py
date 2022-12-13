#  Multi-class Classification
# Neural Networks are often used to classify data. Examples are neural networks:
# - take in photos and classify subjects in the photos as {dog,cat,horse,other}
# - take in a sentence and classify the 'parts of speech' of its elements: {noun, verb, adjective etc..}  

# A network of this type will have multiple units in its final layer.
#  Each output is associated with a category. When an input example is applied to the network, 
# the output with the highest value is the category predicted. If the output is applied to a softmax function, 
# the output of the softmax will provide probabilities of the input being in each category. 

# In this lab you will see an example of building a multiclass network in Tensorflow. 
# We will then take a look at how the neural network makes its predictions.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
np.set_printoptions(precision=2)
from lab_utils_multiclass_TF import *
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

## Data set
# make 4-class dataset for classification
classes = 4
m = 100
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
std = 1.0
X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=std,random_state=30)


## Model
# Below is an example of how to construct this network in Tensorflow. Notice the output layer uses a `linear` rather than a `softmax` activation.
#  While it is possible to include the softmax in the output layer, it is more numerically stable if linear outputs are passed to the loss function
#  during training. If the model is used to predict probabilities, the softmax can be applied at that point

tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        Dense(2, activation = 'relu',   name = "L1"),
        Dense(4, activation = 'linear', name = "L2")
    ]
)

# The statements below compile and train the network. 
# Setting `from_logits=True` as an argument to the loss function specifies 
# that the output activation was linear rather than a softmax.

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

model.fit(
    X_train,y_train,
    epochs=200
)

# In this exercise, you will use a neural network to recognize ten handwritten digits, 0-9.
#  This is a multiclass classification task where one of n choices is selected.
#  Automated handwritten digit recognition is widely used today - from recognizing zip codes (postal codes)
#  on mail envelopes to recognizing amounts written on bank checks. 


### 4.2 Dataset

# You will start by loading the dataset for this task. 
# - The `load_data()` function shown below loads the data into variables `X` and `y`


# - The data set contains 5000 training examples of handwritten digits $^1$.  

#     - Each training example is a 20-pixel x 20-pixel grayscale image of the digit. 
#         - Each pixel is represented by a floating-point number indicating the grayscale intensity at that location. 
#         - The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector. 
#         - Each training examples becomes a single row in our data matrix `X`. 
#         - This gives us a 5000 x 400 matrix `X` where every row is a training example of a handwritten digit image.

# $$X = 
# \left(\begin{array}{cc} 
# --- (x^{(1)}) --- \\
# --- (x^{(2)}) --- \\
# \vdots \\ 
# --- (x^{(m)}) --- 
# \end{array}\right)$$ 

# - The second part of the training set is a 5000 x 1 dimensional vector `y` that contains labels for the training set
#     - `y = 0` if the image is of the digit `0`, `y = 4` if the image is of the digit `4` and so on.

# $^1$<sub> This is a subset of the MNIST handwritten digit dataset (http://yann.lecun.com/exdb/mnist/)</sub>

# load dataset
X, y = load_data()



## Model representation

# The neural network you will use in this assignment is shown in the figure below. 
# - This has two dense layers with ReLU activations followed by an output layer with a linear activation. 
    # - Recall that our inputs are pixel values of digit images.
    # - Since the images are of size $20\times20$, this gives us $400$ inputs  
    

# - The parameters have dimensions that are sized for a neural network with $25$ units in layer 1, $15$ units in layer 2 and $10$ output units in layer 3, one for each digit.
# 
    # - Recall that the dimensions of these parameters is determined as follows:
        # - If network has $s_{in}$ units in a layer and $s_{out}$ units in the next layer, then 
            # - $W$ will be of dimension $s_{in} \times s_{out}$.
            # - $b$ will be a vector with $s_{out}$ elements
  
    # - Therefore, the shapes of `W`, and `b`,  are 
        # - layer1: The shape of `W1` is (400, 25) and the shape of `b1` is (25,)
        # - layer2: The shape of `W2` is (25, 15) and the shape of `b2` is: (15,)
        # - layer3: The shape of `W3` is (15, 10) and the shape of `b3` is: (10,)
# >**Note:** The bias vector `b` could be represented as a 1-D (n,) or 2-D (n,1) array. Tensorflow utilizes a 1-D representation and this lab will maintain that convention: 
                   
# Implementation 

tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [               
        Dense(400, activation = 'relu', name ="L1"),
        Dense(25, activation = 'relu', name ="L2"),
        Dense(15, activation = 'linear', name = "L3")
    ], name = "my_model" 
)

# Complie the model 
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01)
)

# Train the model 
model.fit(
    X,y,
    epochs=50
)

# Plot model report
model.summary()
