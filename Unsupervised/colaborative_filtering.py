## Import libraries 
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import glob 
import os

## Notation
# |General <br />  Notation  | Description| Python (if any) |
# |:-------------|:------------------------------------------------------------||
# | $r(i,j)$     | scalar; = 1  if user j rated movie i  = 0  otherwise             ||
# | $y(i,j)$     | scalar; = rating given by user j on movie  i    (if r(i,j) = 1 is defined) ||
# |$\mathbf{w}^{(j)}$ | vector; parameters for user j ||
# |$b^{(j)}$     |  scalar; parameter for user j ||
# | $\mathbf{x}^{(i)}$ |   vector; feature ratings for movie i        ||     
# | $n_u$        | number of users |num_users|
# | $n_m$        | number of movies | num_movies |
# | $n$          | number of features | num_features                    |
# | $\mathbf{X}$ |  matrix of vectors $\mathbf{x}^{(i)}$         | X |
# | $\mathbf{W}$ |  matrix of vectors $\mathbf{w}^{(j)}$         | W |
# | $\mathbf{b}$ |  vector of bias parameters $b^{(j)}$ | b |
# | $\mathbf{R}$ | matrix of elements $r(i,j)$                    | R |



## 2 - Recommender Systems <img align="left" src="./images/film_rating.png" style=" width:40px;  " >
# In this lab, you will implement the collaborative filtering learning algorithm and apply it to a dataset of movie ratings.
# The goal of a collaborative filtering recommender system is to generate two vectors: 
# For each user;
# a 'parameter vector' that embodies the movie tastes of a user. 
# For each movie:
#  a feature vector of the same size which embodies some description of the movie. 

# The dot product of the two vectors plus the bias term should produce an estimate of 
# the rating the user might give to that movie.


# Existing ratings are provided in matrix form as shown.
#  $Y$ contains ratings; 0.5 to 5 inclusive in 0.5 steps. 
# 0 if the movie has not been rated. $R$ has a 1 where movies have been rated. 
# Movies are in rows, users in columns. 
# Each user has a parameter vector $w^{user}$ and bias. Each movie has a feature vector $x^{movie}$.
#  These vectors are simultaneously learned by using the existing user/movie ratings as training data. 
# One training example is shown above: $\mathbf{w}^{(1)} \cdot \mathbf{x}^{(1)} + b^{(1)} = 4$. 
# It is worth noting that the feature vector $x^{movie}$ must satisfy all the users while the user vector $w^{user}$
#  must satisfy all the movies. 


# Load MovieLens "ml-latest-small" dataset
# https://grouplens.org/datasets/movielens/latest/
# 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users.
# Last updated 9/2018.
# 1,000 movies rated by 700 users

# Load a csv from a folder
data_path = "Unsupervised/data/recomender_systems/colaborative_filtering"
csv_files = glob.glob(os.path.join(data_path,"*.csv"))


W = pd.read_csv(csv_files[0], header=None).to_numpy()
R = pd.read_csv(csv_files[1], header=None).to_numpy()
X = pd.read_csv(csv_files[2], header=None).to_numpy()
b = pd.read_csv(csv_files[3], header=None).to_numpy()
Y = pd.read_csv(csv_files[5], header=None).to_numpy()

num_features = X.shape[1]
num_movies = X.shape[0]
num_users = X.shape[1]

print("Y", Y.shape, "R", R.shape)
print("X", X.shape)
print("W", W.shape)
print("b", b.shape)
print("num_features", num_features)
print("num_movies",   num_movies)
print("num_users",    num_users)

#  From the matrix, we can compute statistics like average rating.
# tsmean =  np.mean(Y[0, R[0, :].astype(bool)])
# print(f"Average rating for movie 1 : {tsmean:0.3f} / 5" )



## 4 - Collaborative filtering learning algorithm <img align="left" src="./images/film_filter.png"     style=" width:40px;  " >

# Now, you will begin implementing the collaborative filtering learning
# algorithm. You will start by implementing the objective function. 

# The collaborative filtering algorithm in the setting of movie
# recommendations considers a set of $n$-dimensional parameter vectors
# $\mathbf{x}^{(0)},...,\mathbf{x}^{(n_m-1)}$, $\mathbf{w}^{(0)},...,\mathbf{w}^{(n_u-1)}$ and $b^{(0)},...,b^{(n_u-1)}$, 
# where the model predicts the rating for movie $i$ by user $j$ as
# $y^{(i,j)} = \mathbf{w}^{(j)}\cdot \mathbf{x}^{(i)} + b^{(j)}$ . Given a dataset that consists of
# a set of ratings produced by some users on some movies, you wish to
# learn the parameter vectors $\mathbf{x}^{(0)},...,\mathbf{x}^{(n_m-1)},
# \mathbf{w}^{(0)},...,\mathbf{w}^{(n_u-1)}$  and $b^{(0)},...,b^{(n_u-1)}$ that produce the best fit (minimizes
# the squared error).


### 4.1 Collaborative filtering cost function

# The collaborative filtering cost function is given by
# $$J({\mathbf{x}^{(0)},...,\mathbf{x}^{(n_m-1)},\mathbf{w}^{(0)},b^{(0)},...,\mathbf{w}^{(n_u-1)},b^{(n_u-1)}})= \frac{1}{2}\sum_{(i,j):r(i,j)=1}(\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - y^{(i,j)})^2
# +\underbrace{
# \frac{\lambda}{2}
# \sum_{j=0}^{n_u-1}\sum_{k=0}^{n-1}(\mathbf{w}^{(j)}_k)^2
# + \frac{\lambda}{2}\sum_{i=0}^{n_m-1}\sum_{k=0}^{n-1}(\mathbf{x}_k^{(i)})^2
# }_{regularization}
# \tag{1}$$
# The first summation in (1) is "for all $i$, $j$ where $r(i,j)$ equals $1$" and could be written:
# 
# $$
# = \frac{1}{2}\sum_{j=0}^{n_u-1} \sum_{i=0}^{n_m-1}r(i,j)*(\mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - y^{(i,j)})^2
# +\text{regularization}
# $$

# You should now write cofiCostFunc (collaborative filtering cost function) to return this cost.

def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    nm, nu = Y.shape
    nf = X.shape[1]
    J = 0
    reg_feature = 0
    reg_weights = 0

    
    # Compute i,j such that R(i,j) = 1
    rated_indexes = R == 1
    
    for user in range(nu):
        for movie in range(nm):
            if R[movie, user] == 1:
                J_ij=.5*(np.dot(W[user,:], X[movie,:]) + b[0,user] -Y[movie,user])**2
                J += J_ij
    
    for movie in range(nm):
        for feature in range(nf):
            reg_feature += .5 * lambda_ * X[movie,feature]**2

    for user in range(nu):
        for feature in range(nf):
            reg_weights += .5 * lambda_ * W[user,feature]**2

    J += reg_weights + reg_feature
    
    return J

# **Vectorized Implementation**

# It is important to create a vectorized implementation to compute $J$, 
# since it will later be called many times during optimization.
#  The linear algebra utilized is not the focus of this series, so the implementation is provided. 
# If you are an expert in linear algebra, feel free to create your version without referencing the code below. 

# Run the code below and verify that it produces the same results as the non-vectorized version.

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J


# Reduce the data set size so that this runs faster
num_users_r = 4
num_movies_r = 5 
num_features_r = 3

X_r = X[:num_movies_r, :num_features_r]
W_r = W[:num_users_r,  :num_features_r]
b_r = b[0, :num_users_r].reshape(1,-1)
Y_r = Y[:num_movies_r, :num_users_r]
R_r = R[:num_movies_r, :num_users_r]

print(X_r.shape)
print(W_r.shape)
print(b_r.shape)
print(Y_r.shape)
print(R_r.shape)


# Evaluate cost function
J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 0);
print(f"Cost: {J:0.2f}")


# Evaluate cost function
Jv = cofi_cost_func_v(X_r, W_r, b_r, Y_r, R_r, 0);
print(f"Cost vect: {Jv:0.2f}")

# Evaluate cost function with regularization 
J = cofi_cost_func_v(X_r, W_r, b_r, Y_r, R_r, 1.5);
print(f"Cost (with regularization): {J:0.2f}")


# Create the tensofrflow varialbles
#  Useful Values
num_movies, num_users = Y.shape
num_features = 100

# Set Initial Parameters (W, X), use tf.Variable to track these variables
tf.random.set_seed(1234) # for consistent results
W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-1)


iterations = 1000
lambda_ = 1
for iter in range(iterations):
    # Use TensorFlow’s GradientTape
    # to record the operations used to compute the cost 
    with tf.GradientTape() as tape:

        # Compute the cost (forward pass included in cost)
        cost_value = cofi_cost_func_v(X, W, b, Y, R, lambda_)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss
    grads = tape.gradient( cost_value, [X,W,b] )

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients( zip(grads, [X,W,b]) )

    # Log periodically.
    if iter % 20 == 0:
        print(f"Training loss at iteration {iter}: {cost_value:0.1f}")


# Make a prediction using trained weights and biases
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
