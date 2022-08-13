#

# Examples of classification problems are things like: identifying email as Spam or
#  Not Spam or determining if a tumor is malignant or benign. In particular, these are examples
#  of *binary* classification where there are two possible outcomes.  Outcomes can be  described
#  in pairs of 'positive'/'negative' such as 'yes'/'no, 'true'/'false' or '1'/'0'. 

# Plots of classification data sets often use symbols to indicate the outcome of an example. 

# Problem Statement
import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
# from lab_utils_common import dlc, plot_data
# from plt_one_addpt_onclick import plt_one_addpt_onclick
# plt.style.use('./../notes/deeplearning.mplstyle')

# Data set

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])


# Plots

pos = y_train == 1
neg = y_train == 0

fig,ax = plt.subplots(1,2,figsize=(8,3))
#plot 1, single variable
ax[0].scatter(x_train[pos], y_train[pos], marker='x', s=80, c = 'red', label="y=1")
# ax[0].scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
            #   edgecolors=dlc["dlblue"],lw=3)

ax[0].set_ylim(-0.08,1.1)
ax[0].set_ylabel('y', fontsize=12)
ax[0].set_xlabel('x', fontsize=12)
ax[0].set_title('one variable plot')
ax[0].legend()

#plot 2, two variables
plt.plot(X_train2, y_train2)
# ax[1].axis([0, 4, 0, 4])
# ax[1].set_ylabel('$x_1$', fontsize=12)
# ax[1].set_xlabel('$x_0$', fontsize=12)
# ax[1].set_title('two variable plot')
# ax[1].legend()
# plt.tight_layout()
# plt.show()


# In the previous week, you applied linear regression to build a prediction model. Let's try that approach here using the simple example that was described in the lecture. The model will predict if a tumor is benign or malignant based on tumor size.  Try the following:
# - Click on 'Run Linear Regression' to find the best linear regression model for the given data.
#     - Note the resulting linear model does **not** match the data well. 
# One option to improve the results is to apply a *threshold*. 
# - Tick the box on the 'Toggle 0.5 threshold' to show the predictions if a threshold is applied.
#     - These predictions look good, the predictions match the data
# - *Important*: Now, add further 'malignant' data points on the far right, in the large tumor size range (near 10), and re-run linear regression.
#     - Now, the model predicts the larger tumor, but data point at x=3 is being incorrectly predicted!
# - to clear/renew the plot, rerun the cell containing the plot command.

w_in = np.zeros((1))
b_in = 0
# plt.close('all') 
# addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=False)




## Sigmoid function

# The formula for a sigmoid function is as follows -  

# $g(z) = \frac{1}{1+e^{-z}}\tag{1}$

# In the case of logistic regression, z (the input to the sigmoid function), is the output of a linear regression model. 
# - In the case of a single example, $z$ is scalar.
# - in the case of multiple examples, $z$ may be a vector consisting of $m$ values, one for each example. 
# - The implementation of the sigmoid function should cover both of these potential input formats.
# Let's implement this in Python.


# Input is an array. 
input_array = np.array([1,2,3])
exp_array = np.exp(input_array)

print("Input to exp:", input_array)
print("Output of exp:", exp_array)

# Input is a single number
input_val = 1  
exp_val = np.exp(input_val)

print("Input to exp:", input_val)
print("Output of exp:", exp_val)

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """

    g = 1/(1+np.exp(-z))
   
    return g

    # Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)

# Use the function implemented above to get the sigmoid values
y = sigmoid(z_tmp)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])

# Plot z vs sigmoid(z)
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
# draw_vthresh(ax,0)



##  Logistic regression

# A logistic regression model applies the sigmoid to the familiar linear regression model 
# as shown below:

# $$ f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot \mathbf{x}^{(i)} + b ) \tag{2} $$ 

#   where

#   $g(z) = \frac{1}{1+e^{-z}}\tag{3}$
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0

##  Decision boundary 
# Define data set
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 


# Plot data
fig,ax = plt.subplots(1,1,figsize=(4,4))
plt.plot(X, y)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.show()


## Logistic regression model

# * Suppose you'd like to train a logistic regression model on this data which has the form   
#   $f(x) = g(w_0x_0+w_1x_1 + b)$
#   where $g(z) = \frac{1}{1+e^{-z}}$, which is the sigmoid function

# * Let's say that you trained the model and get the parameters as $b = -3, w_0 = 1, w_1 = 1$. That is,
#   $f(x) = g(x_0+x_1-3)$
#   (You'll learn how to fit these parameters to the data further in the course)
  
  
# Let's try to understand what this trained model is predicting by plotting its decision boundary

### Refresher on logistic regression and decision boundary

# * Recall that for logistic regression, the model is represented as 
#   $$f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot \mathbf{x}^{(i)} + b) \tag{1}$$
#   where $g(z)$ is known as the sigmoid function and it maps all input values to values between 0 and 1:
#   $g(z) = \frac{1}{1+e^{-z}}\tag{2}$
#   and $\mathbf{w} \cdot \mathbf{x}$ is the vector dot product:
#   $$\mathbf{w} \cdot \mathbf{x} = w_0 x_0 + w_1 x_1$$
  
#  * We interpret the output of the model ($f_{\mathbf{w},b}(x)$) as the probability that $y=1$ given $\mathbf{x}$ and parameterized by $\mathbf{w}$ and $b$.
# * Therefore, to get a final prediction ($y=0$ or $y=1$) from the logistic regression model, we can use the following heuristic -

#   if $f_{\mathbf{w},b}(x) >= 0.5$, predict $y=1$
#   if $f_{\mathbf{w},b}(x) < 0.5$, predict $y=0$
  
# * Let's plot the sigmoid function to see where $g(z) >= 0.5$

# Plot sigmoid(z) over a range of values from -10 to 10
z = np.arange(-10,11)

fig,ax = plt.subplots(1,1,figsize=(5,3))
# Plot z vs sigmoid(z)
ax.plot(z, sigmoid(z), c="b")


## Logistic loss cost function

# Since the classic (y_predict - y_m)^2 we use 

# Logistic Regression uses a loss function more suited to the task of categorization where the target is 0 or 1 rather than any number. 

# >**Definition Note:**   In this course, these definitions are used:  
# **Loss** is a measure of the difference of a single example to its target value while the  
# **Cost** is a measure of the losses over the training set


# This is defined: 
# * $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ is the cost for a single data point, which is:

# \begin{equation}
#   loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = \begin{cases}
#     - \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) & \text{if $y^{(i)}=1$}\\
#     - \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) & \text{if $y^{(i)}=0$}
#   \end{cases}
# \end{equation}


# *  $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ is the model's prediction,
#  while $y^{(i)}$ is the target value.

# *  $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot\mathbf{x}^{(i)}+b)$ where function $g$ 
# is the sigmoid function.

# The defining feature of this loss function is the fact that it uses two separate curves.
#  One for the case when the target is zero or ($y=0$) and another for when the target is one ($y=1$).
#  Combined, these curves provide the behavior useful for a loss function, namely, being zero when the prediction 
# matches the target and rapidly increasing in value as the prediction differs from the target.


# Let's implement the cost function: 

fig,ax = plt.subplots(1,1,figsize=(4,4))
plt.plot(X_train, y_train, ax)

# Set both axes to be from 0-4
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()

# Define logistic cost
def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost



w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))


import matplotlib.pyplot as plt

# Choose values between 0 and 6
x0 = np.arange(0,6)

# Plot the two decision boundaries
x1 = 3 - x0
x1_other = 4 - x0

fig,ax = plt.subplots(1, 1, figsize=(4,4))
# Plot the decision boundary
ax.plot(x0,x1, c=dlc["dlblue"], label="$b$=-3")
ax.plot(x0,x1_other, c=dlc["dlmagenta"], label="$b$=-4")
ax.axis([0, 4, 0, 4])

# Plot the original data
plot_data(X_train,y_train,ax)
ax.axis([0, 4, 0, 4])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.legend(loc="upper right")
plt.title("Decision Boundary")
plt.show()


w_array1 = np.array([1,1])
b_1 = -3
w_array2 = np.array([1,1])
b_2 = -4

print("Cost for b = -3 : ", compute_cost_logistic(X_train, y_train, w_array1, b_1))
print("Cost for b = -4 : ", compute_cost_logistic(X_train, y_train, w_array2, b_2))


# **Expected output**

# Cost for b = -3 :  0.3668667864055175

# Cost for b = -4 :  0.5036808636748461