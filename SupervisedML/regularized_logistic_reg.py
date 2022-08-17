## 3 - Regularized Logistic Regression

# In this part of the exercise, you will implement regularized logistic regression to predict whether 
# microchips from a fabrication plant passes quality assurance (QA). During QA, each microchip goes
#  through various tests to ensure it is functioning correctly. 

### Load libraries 

import numpy as np
import matplotlib.pyplot as plt
# from utils import *
import copy
import math

### 3.1 Problem Statement

# Suppose you are the product manager of the factory and you have the test results for some microchips on
#  two different tests. 
# - From these two tests, you would like to determine whether the microchips should be accepted or rejected. 
# - Tot help you make the decision, you have a dataset of test results on past microchips, from which you can
#  build a logistic regression model.

### 3.2 Loading and visualizing the data

# Similar to previous parts of this exercise, let's start by loading the dataset for this task and visualizing it. 

# - The `load_dataset()` function shown below loads the data into variables `X_train` and `y_train`
#   - `X_train` contains the test results for the microchips from two tests
#   - `y_train` contains the results of the QA  
    #   - `y_train = 1` if the microchip was accepted 
    #   - `y_train = 0` if the microchip was rejected 
#   - Both `X_train` and `y_train` are numpy arrays.

# load dataset
X_train, y_train = load_data("data/ex2data2.txt")

# print X_train
print("X_train:", X_train[:5])
print("Type of X_train:",type(X_train))

# print y_train
print("y_train:", y_train[:5])
print("Type of y_train:",type(y_train))

# Plot examples
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")

# Set the y-axis label
plt.ylabel('Microchip Test 2') 
# Set the x-axis label
plt.xlabel('Microchip Test 1') 
plt.legend(loc="upper right")
plt.show()

### 3.3 Feature mapping

# One way to fit the data better is to create more features from each data point.
#  In the provided function `map_feature`, we will map the features into all polynomial terms 
# of $x_1$ and $x_2$ up to the sixth power.

# $$\mathrm{map\_feature}(x) = 
# \left[\begin{array}{c}
# x_1\\
# x_2\\
# x_1^2\\
# x_1 x_2\\
# x_2^2\\
# x_1^3\\
# \vdots\\
# x_1 x_2^5\\
# x_2^6\end{array}\right]$$

# As a result of this mapping, our vector of two features (the scores on two QA tests) has been transformed into a 27-dimensional vector. 

# - A logistic regression classifier trained on this higher-dimension feature vector will have a more complex decision boundary and will be nonlinear when drawn in our 2-dimensional plot. 
# - We have provided the `map_feature` function for you in utils.py. 

print("Original shape of data:", X_train.shape)

mapped_X =  map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)

print("X_train[0]:", X_train[0])
print("mapped X_train[0]:", mapped_X[0])


# While the feature mapping allows us to build a more expressive classifier, it is also more susceptible to overfitting. 
# In the next parts of the exercise, you will implement regularized logistic regression to fit the data and also see for
#  yourself how regularization can help combat the overfitting problem.


### 3.4 Cost function for regularized logistic regression

# In this part, you will implement the cost function for regularized logistic regression.

# Recall that for regularized logistic regression, the cost function is of the form
# $$J(\mathbf{w},b) =
#  \frac{1}{m}  \sum_{i=0}^{m-1} \left[ -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \right] + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$

# Compare this to the cost function without regularization (which you implemented above), which is of the form 

# $$ J(\mathbf{w}.b) 
# = \frac{1}{m}\sum_{i=0}^{m-1} \left[ (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)\right]$$

# The difference is the regularization term, which is $$\frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$ 
# Note that the $b$ parameter is not regularized.


### Exercise 5

# Please complete the `compute_cost_reg` function below to calculate the following term for each element in $w$ 
# $$\frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$

# The starter code then adds this to the cost without regularization (which you computed above in `compute_cost`)
#  to calculate the cost with regularization.
# UNQ_C5
def compute_cost_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X : (array_like Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : (array_like Shape (n,)) Values of bias parameter of the model
      lambda_ : (scalar, float)    Controls amount of regularization
    Returns:
      total_cost: (scalar)         cost 
    """

    m, n = X.shape
    
    # Calls the compute_cost function that you implemented above
    cost_without_reg = compute_cost(X, y, w, b) 
    
    # You need to calculate this value
    reg_cost = np.sum(w**2)
    
    # Add the regularization cost to get the total cost
    total_cost = cost_without_reg + (lambda_/(2 * m)) * reg_cost

    return total_cost


X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print("Regularized cost :", cost)


# UNQ_C6
def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X : (ndarray Shape (m,n))   variable such as house size 
      y : (ndarray Shape (m,))    actual value 
      w : (ndarray Shape (n,))    values of parameters of the model      
      b : (scalar)                value of parameter of the model  
      lambda_ : (scalar,float)    regularization constant
    Returns
      dj_db: (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw: (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

    """
    m, n = X.shape
    
    dj_db, dj_dw = compute_gradient(X, y, w, b)

    for j in range(n):
        dj_dw_j_reg = lambda_ / m * w[j] 
        dj_dw[j] = dj_dw[j] + dj_dw_j_reg  
        
    return dj_db, dj_dw


# Initialize fitting parameters
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1])-0.5
initial_b = 1.

# Set regularization parameter lambda_ to 1 (you can try varying this)
lambda_ = 0.01;                                          
# Some gradient descent settings
iterations = 10000
alpha = 0.01

w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b, 
                                    compute_cost_reg, compute_gradient_reg, 
                                    alpha, iterations, lambda_)