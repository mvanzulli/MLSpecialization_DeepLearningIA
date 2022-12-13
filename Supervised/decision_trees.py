#  Packages 

# First, let's run the cell below to import all the packages that you will need during this assignment.
# - [numpy](https://www.numpy.org) is the fundamental package for working with matrices in Python.
# - [matplotlib](https://matplotlib.org) is a famous library to plot graphs in Python.
# - ``utils.py`` contains helper functions for this assignment. You do not need to modify code in this file.

import numpy as np
import matplotlib.pyplot as plt

# 2 -  Problem Statement

# Suppose you are starting a company that grows and sells wild mushrooms. 
# - Since not all mushrooms are edible, you'd like to be able to tell whether
#  a given mushroom is edible or poisonous based on it's physical attributes
# - You have some existing data that you can use for this task. 
# 
# Can you use the data to help you identify which mushrooms can be sold safely? 

# Note: The dataset used is for illustrative purposes only. It is not meant to be 
# a guide on identifying edible mushrooms.


# You will start by loading the dataset for this task. The dataset you have collected is as follows:

# | Cap Color | Stalk Shape | Solitary | Edible |
# |:---------:|:-----------:|:--------:|:------:|
# |   Brown   |   Tapering  |    Yes   |    1   |
# |   Brown   |  Enlarging  |    Yes   |    1   |
# |   Brown   |  Enlarging  |    No    |    0   |
# |   Brown   |  Enlarging  |    No    |    0   |
# |   Brown   |   Tapering  |    Yes   |    1   |
# |    Red    |   Tapering  |    Yes   |    0   |
# |    Red    |  Enlarging  |    No    |    0   |
# |   Brown   |  Enlarging  |    Yes   |    1   |
# |    Red    |   Tapering  |    No    |    1   |
# |   Brown   |  Enlarging  |    No    |    0   |


# -  You have 10 examples of mushrooms. For each example, you have
    # - Three features
        # - Cap Color (`Brown` or `Red`),
        # - Stalk Shape (`Tapering (as in \/)` or `Enlarging (as in /\)`), and
        # - Solitary (`Yes` or `No`)
    # - Label
        # - Edible (`1` indicating yes or `0` indicating poisonous) 

### 3.1 One hot encoded dataset
# For ease of implementation, we have one-hot encoded the features (turned them into 0 or 1 valued features)
# 
#| Brown Cap | Tapering Stalk Shape | Solitary | Edible |
#|:---------:|:--------------------:|:--------:|:------:|
#|     1     |           1          |     1    |    1   |
#|     1     |           0          |     1    |    1   |
#|     1     |           0          |     0    |    0   |
#|     1     |           0          |     0    |    0   |
#|     1     |           1          |     1    |    1   |
#|     0     |           1          |     1    |    0   |
#|     0     |           0          |     0    |    0   |
#|     1     |           0          |     1    |    1   |
#|     0     |           1          |     0    |    1   |
#|     1     |           0          |     0    |    0   |


# Therefore,
# - `X_train` contains three features for each example 
    # -t Brown Color (A value of `1` indicates "Brown" cap color and `0` indicates "Red" cap color)
    # - Tapering Shape (A value of `1` indicates "Tapering Stalk Shape" and `0` indicates "Enlarging" stalk shape)
    # - Solitary  (A value of `1` indicates "Yes" and `0` indicates "No")
# 
# - `y_train` is whether the mushroom is edible 
    # - `y = 1` indicates edible
    # - `y = 0` indicates poisonous

X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])

print("First few elements of X_train:\n", X_train[:5])
print("Type of X_train:",type(X_train))
print ('The shape of X_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(X_train))

## 4 - Decision Tree Refresher


# - Recall that the steps for building a decision tree are as follows:
    # - Start with all examples at the root node
    # - Calculate information gain for splitting on all possible features, and pick the one with the highest information gain
    # - Split dataset according to the selected feature, and create left and right branches of the tree
    # - Keep repeating splitting process until stopping criteria is met
  
  
# - In this lab, you'll implement the following functions, which will let you split a node into left and right branches using the feature with the highest information gain
    # - Calculate the entropy at a node 
    # - Split the dataset at a node into left and right branches based on a given feature
    # - Calculate the information gain from splitting on a given feature
    # - Choose the feature that maximizes information gain
    
# - We'll then use the helper functions you've implemented to build a decision tree by repeating the splitting process until the stopping criteria is met 
    # - For this lab, the stopping criteria we've chosen is setting a maximum depth of 2


    ### 4.1  Calculate entropy

# First, you'll write a helper function called `compute_entropy` that computes the entropy (measure of impurity) at a node. 
# - The function takes in a numpy array (`y`) that indicates whether the examples in that node are edible (`1`) or poisonous(`0`) 

# Complete the `compute_entropy()` function below to:
# * Compute $p_1$, which is the fraction of examples that are edible (i.e. have value = `1` in `y`)
# * The entropy is then calculated as 

# $$H(p_1) = -p_1 \text{log}_2(p_1) - (1- p_1) \text{log}_2(1- p_1)$$
# * Note 
    # * The log is calculated with base $2$
    # * For implementation purposes, $0\text{log}_2(0) = 0$. That is, if `p_1 = 0` or `p_1 = 1`, set the entropy to `0`
    # * Make sure to check that the data at a node is not empty (i.e. `len(y) != 0`). Return `0` if it is

def compute_entropy(y):
    """
    Computes the entropy for 
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
       
    Returns:
        entropy (float): Entropy at that node
        
    """
    # You need to return the following variables correctly
    entropy = 0.
    
    # Check length
    m = len(y)
    if m == 0:
        return entropy 

    # Compute portion 
    num_eddible = sum(y)
    num_poiss = len(y) - num_eddible
    p_e = num_eddible / m
    p_p = num_poiss / m

    # Check sum of ps
    assert p_e + p_p == 1, "The probablitie of p_e and p_p must sum 1"
    
    # Compute entropy
    if (p_p == 0) or (p_e == 0):
        entropy = 0 
    else:
        entropy = -p_e*np.log2(p_e) - p_p * np.log2(p_p)
    
    return entropy

# 4.2  Split dataset

# Next, you'll write a helper function called `split_dataset` that takes in the data at a node and a
#  feature to split on and splits it into left and right branches. Later in the lab, you'll implement 
# code to calculate how good the split is.

# - The function takes in the training data, the list of indices of data points at that node, along 
# with the feature to split on. 
# - It splits the data and returns the subset of indices at the left and the right branch.
# - For example, say we're starting at the root node (so `node_indices = [0,1,2,3,4,5,6,7,8,9]`), and we chose
#  to split on feature `0`, which is whether or not the example has a brown cap. 
    # - The output of the function is then, `left_indices = [0,1,2,3,4,7,9]` (data points with brown cap) and 
    # `right_indices = [5,6,8]` (data points without a brown cap)

    ### Exercise 2

# Please complete the `split_dataset()` function shown below

# - For each index in `node_indices`
    # - If the value of `X` at that index for that feature is `1`, add the index to `left_indices`
    # - If the value of `X` at that index for that feature is `0`, add the index to `right_indices`

def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on
    
    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """
    
    # You need to return the following variables correctly
    left_indices = []
    right_indices = []
    
    for index in node_indices: 
        # push to left is the selected feature is present
        X[index, feature] == 1 and left_indices.append(index) 
        # push to right is the selected feature is not present
        X[index, feature] == 0 and right_indices.append(index) 

    return left_indices, right_indices


# Test function 
root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Feel free to play around with these variables

feature = 0

left_indices, right_indices = split_dataset(X_train, root_indices, feature)

print("Left indices: ", left_indices)
print("Right indices: ", right_indices)


# 4.3  Calculate information gain

# Next, you'll write a function called `information_gain` that takes in the training data, the indices at a node and a feature to split on and returns the information gain from the split.

### Exercise 3

# Please complete the `compute_information_gain()` function shown below to compute

# $$\text{Information Gain} = H(p_1^\text{node})- (w^{\text{left}}H(p_1^\text{left}) + w^{\text{right}}H(p_1^\text{right}))$$

# where 
# - $H(p_1^\text{node})$ is entropy at the node 
# - $H(p_1^\text{left})$ and $H(p_1^\text{right})$ are the entropies at the left and the right branches resulting from the split
# - $w^{\text{left}}$ and $w^{\text{right}}$ are the proportion of examples at the left and right branch, respectively

# Note:
# - You can use the `compute_entropy()` function that you implemented above to calculate the entropy
# - We've provided some starter code that uses the `split_dataset()` function you implemented above to split the dataset 

def compute_information_gain(X, y, node_indices, feature):
    
    """
    Compute the information of splitting the node on a given feature
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
   
    Returns:
        cost (float):        Cost computed
    
    """    
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    # You need to return the following variables correctly
    information_gain = 0
    
    # number of samples 
    m = len(y_node)
    # H(p1)
    H_node = compute_entropy(y_node)
    # H(left)
    H_left = compute_entropy(y_left)
    # H(right)
    H_right = compute_entropy(y_right)
    # w left 
    w_left = len(y_left) / m
    # w right 
    w_right = len(y_right) / m

    information_gain = H_node - (w_left * H_left + w_right * H_right )
    
    return information_gain

info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
print("Information Gain from splitting the root on brown cap: ", info_gain0)

info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)

info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
print("Information Gain from splitting the root on solitary: ", info_gain2)

# Assert expected results 
assert info_gain0 == 0.034851554559677034
assert info_gain1 == 0.12451124978365313
assert info_gain2 == 0.2780719051126377


def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature and threshold value
    to split the node data 
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """    
    
    # Some useful variables
    num_features = X.shape[1]
    
    # You need to return the following variables correctly
    best_feature = -1
    # maximum information gain for the node 
    max_info_gain = 0.
    for current_feature in range(num_features):
        
        # Compute information gain for the current feature 
        info_gain_feature = compute_information_gain(X, y, node_indices, current_feature)

        # If the current feature is better than the previous pick it up:  
        if info_gain_feature > max_info_gain:
            best_feature = current_feature
            max_info_gain = info_gain_feature
            
    return best_feature

best_feature = get_best_split(X_train, y_train, root_indices)
print("Best feature to split on: %d" % best_feature)

assert best_feature == 2

# 5 - Building the tree

# In this section, we use the functions you implemented above to generate a decision
#  tree by successively picking the best feature to split on until we reach the stopping 
# criteria (maximum depth is 2).
# 
tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree. 
        current_depth (int):    Current depth. Parameter used during recursive call.
   
    """ 

    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices) 
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    
    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)

build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)