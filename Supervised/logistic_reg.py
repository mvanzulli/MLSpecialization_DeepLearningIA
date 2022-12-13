# Problem Statement 
# Suppose that you are the administrator of a university department and you want to determine each applicant’s chance of admission based on their results on two exams. 
# * You have historical data from previous applicants that you can use as a training set for logistic regression. 
# * For each training example, you have the applicant’s scores on two exams and the admissions decision. 
# * Your task is to build a classification model that estimates an applicant’s probability of admission based on the scores from those two exams. 


# Import libraries 

import numpy as np
import matplotlib.pyplot as plt
# from utils import *
import copy
import math


# Load dataset 

# You will start by loading the dataset for this task. 
# - The `load_dataset()` function shown below loads the data into variables `X_train` and `y_train`
#   - `X_train` contains exam scores on two exams for a student
#   - `y_train` is the admission decision 
#       - `y_train = 1` if the student was admitted 
#       - `y_train = 0` if the student was not admitted 
#   - Both `X_train` and `y_train` are numpy arrays.

# load dataset
X_train, y_train = load_data("data/ex2data1.txt")

print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train:",type(X_train))

print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))

# Another useful way to get familiar with your data is to view its dimensions. Let's print the shape of `X_train` and `y_train` 
# and see how many training examples we have in our dataset.
print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))


# Plot data

# Plot examples
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()


### 2.3  Sigmoid function

# Recall that for logistic regression, the model is represented as

# $$ f_{\mathbf{w},b}(x) = g(\mathbf{w}\cdot \mathbf{x} + b)$$
# where function $g$ is the sigmoid function. The sigmoid function is defined as:

# $$g(z) = \frac{1}{1+e^{-z}}$$

# Let's implement the sigmoid function first, so it can be used by the rest of this assignment.

# <a name='ex-01'></a>
# ### Exercise 1
# Please complete  the `sigmoid` function to calculate

# $$g(z) = \frac{1}{1+e^{-z}}$$

# Note that 
# - `z` is not always a single number, but can also be an array of numbers. 
# - If the input is an array of numbers, we'd like to apply the sigmoid function to each value in the input array.

# If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """
          
    g = 1 / (1 + np.exp(-z))
    return g

print ("sigmoid(0) = " + str(sigmoid(0)))



### 2.4 Cost function for logistic regression

# In this section, you will implement the cost function for logistic regression.

# <a name='ex-02'></a>

### Exercise 2

# Please complete the `compute_cost` function using the equations below.

# Recall that for logistic regression, the cost function is of the form 

# $$ J(\mathbf{w},b) = \frac{1}{m}\sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right] \tag{1}$$

# where
# * m is the number of training examples in the dataset


# * $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ is the cost for a single data point, which is - 

#     $$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \tag{2}$$
    
    
# *  $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ is the model's prediction, while $y^{(i)}$, which is the actual label

# *  $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot \mathbf{x^{(i)}} + b)$ where function $g$ is the sigmoid function.
#     * It might be helpful to first calculate an intermediate variable $z_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x^{(i)}} + b = w_0x^{(i)}_0 + ... + w_{n-1}x^{(i)}_{n-1} + b$ where $n$ is the number of features, before calculating $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(z_{\mathbf{w},b}(\mathbf{x}^{(i)}))$

# Note:
# * As you are doing this, remember that the variables `X_train` and `y_train` are not scalar values but matrices of shape ($m, n$) and ($𝑚$,1) respectively, where  $𝑛$ is the number of features and $𝑚$ is the number of training examples.
# * You can use the sigmoid function that you implemented above for this part.

# If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.

def compute_cost(X, y, w, b, lambda_= 1):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder
    Returns:
      total_cost: (scalar)         cost 
    """
    # Extract shape 
    m, n = X.shape
    
    total_cost = .0
    # Compute cost
    for i in range(m):
        z = np.dot(X[i, :], w) + b
        f_wb = sigmoid(z)
        yi = y[i]
        total_cost += -yi * np.log(f_wb) - (1 - yi) * np.log(1 - f_wb)

    
    
    return total_cost / m

m, n = X_train.shape

# Compute and display cost with w initialized to zeroes
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w (zeros): {:.3f}'.format(cost))


# **Expected Output**:
# <table>
#   <tr>
#     <td> <b>Cost at initial w (zeros)<b></td>
#     <td> 0.693 </td> 
#   </tr>
# </table>


### Exercise 3

# Please complete the `compute_gradient` function to compute $\frac{\partial J(\mathbf{w},b)}{\partial w}$, $\frac{\partial J(\mathbf{w},b)}{\partial b}$ from equations (2) and (3) below.

# $$
# \frac{\partial J(\mathbf{w},b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - \mathbf{y}^{(i)}) \tag{2}
# $$
# $$
# \frac{\partial J(\mathbf{w},b)}{\partial w_j}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - \mathbf{y}^{(i)})x_{j}^{(i)} \tag{3}
# $$
# * m is the number of training examples in the dataset

    
# *  $f_{\mathbf{w},b}(x^{(i)})$ is the model's prediction, while $y^{(i)}$ is the actual label


# - **Note**: While this gradient looks identical to the linear regression gradient, the formula is actually different because linear and logistic regression have different definitions of $f_{\mathbf{w},b}(x)$.

# As before, you can use the sigmoid function that you implemented above and if you get stuck, you can check out the hints presented after the cell below to help you with the implementation.

def compute_gradient(X, y, w, b, lambda_=None): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X : (ndarray Shape (m,n)) variable such as house size 
      y : (array_like Shape (m,1)) actual value 
      w : (array_like Shape (n,1)) values of parameters of the model      
      b : (scalar)                 value of parameter of the model 
      lambda_: unused placeholder.
    Returns
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w. 
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        # y_predict_i
        z_wb = np.dot(X[i, :], w) + b
        f_wb = sigmoid(z_wb)
        # y_train_i
        y_i = y[i]
        
        dj_db_i = f_wb - y_i 
        dj_db += dj_db_i
        
        for j in range(n):
            dj_dw[j] += dj_db_i * X[i,j]
    
    
    dj_dw = np.dot(dj_dw, 1 / m)
    dj_db = dj_db / m
    

    return dj_db, dj_dw  


# Compute and display gradient with w initialized to zeroes
initial_w = np.zeros(n)
initial_b = 0.

dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w (zeros):{dj_db}' )
print(f'dj_dw at initial w (zeros):{dj_dw.tolist()}' )

# **Expected Output**:
# <table>
#   <tr>
#     <td> <b>dj_db at initial w (zeros)<b></td>
#     <td> -0.1 </td> 
#   </tr>
#   <tr>
#     <td> <b>ddj_dw at initial w (zeros):<b></td>
#     <td> [-12.00921658929115, -11.262842205513591] </td> 
#   </tr>
# </table>

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X :    (array_like Shape (m, n)
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)                 Initial value of parameter of the model
      cost_function:                  function to compute cost
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant
      
    Returns:
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing



# Train model 

np.random.seed(1)
intial_w = 0.01 * (np.random.rand(2).reshape(-1,1) - 0.5)
initial_b = -8


# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)


### Exercise 4

# Please complete the `predict` function to produce `1` or `0` predictions given
#  a dataset and a learned parameter vector $w$ and $b$.
# - First you need to compute the prediction from the model $f(x^{(i)}) = g(w \cdot x^{(i)})$ for every example 
#     - You've implemented this before in the parts above
# - We interpret the output of the model ($f(x^{(i)})$) as the probability 
# that $y^{(i)}=1$ given $x^{(i)}$ and parameterized by $w$.
# - Therefore, to get a final prediction ($y^{(i)}=0$ or $y^{(i)}=1$) from the logistic regression model, you can use the following heuristic -

#   if $f(x^{(i)}) >= 0.5$, predict $y^{(i)}=1$
  
#   if $f(x^{(i)}) < 0.5$, predict $y^{(i)}=0$
    
# If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.

# UNQ_C4
# GRADED FUNCTION: predict

def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    
    Args:
    X : (ndarray Shape (m, n))
    w : (array_like Shape (n,))      Parameters of the model
    b : (scalar, float)              Parameter of the model

    Returns:
    p: (ndarray (m,1))
        The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape   
    p = np.zeros(m)
   
    ### START CODE HERE ### 
    # Loop over each example
    for i in range(m):   
        z_wb = .0
        # Loop over each feature
        for j in range(n): 
            # Add the corresponding term to z_wb
            z_wb += X[i,j]*w[j]
        
        # Add bias term 
        z_wb += b
        
        # Calculate the prediction for this example
        f_wb = sigmoid(z_wb)

        # Apply the threshold
        p[i] = 1 if f_wb > .5 else 0
        
    ### END CODE HERE ### 
    return p

#Compute accuracy on our training set
p = predict(X_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))