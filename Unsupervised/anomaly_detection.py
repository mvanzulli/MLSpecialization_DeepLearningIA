## Import libraries 
import numpy as np
import matplotlib.pyplot as plt
import glob

## 2 - Anomaly detection

### 2.1 Problem Statement

# In this exercise, you will implement an anomaly detection algorithm to
# detect anomalous behavior in server computers.

# The dataset contains two features - 
#    * throughput (mb/s) and 
#    * latency (ms) of response of each server.

# While your servers were operating, you collected $m=307$ examples of how 
# they were behaving, 
# and thus have an unlabeled dataset $\{x^{(1)}, \ldots, x^{(m)}\}$. 
# * You suspect that the vast majority of these examples are “normal” 
# (non-anomalous) examples of the servers operating normally, 
# but there might also be some examples of servers acting anomalously 
# within this dataset.

# You will use a Gaussian model to detect anomalous examples in your
# dataset. 
# * You will first start on a 2D dataset that will allow you to visualize 
# what the algorithm is doing.
# * On that dataset you will fit a Gaussian distribution and then
#  find values that have very low probability and hence can be considered anomalies. 
# * After that, you will apply the anomaly detection algorithm to a larger dataset with many dimensions. 

# ### 2.2  Dataset

# You will start by loading the dataset for this task. 
# - The `load_data()` function shown below loads the data into the variables `X_train`, `X_val` and `y_val` 
#     - You will use `X_train` to fit a Gaussian distribution 
#     - You will use `X_val` and `y_val` as a cross validation set to select a threshold and determine anomalous vs normal examples

# Load the dataset

# load data/anomaly .npy files
npfiles = glob.glob('UnsuperivsedML/data/anomaly/*.npy')

## Load .npy data files from the current directory
X_train = np.load(npfiles[1])
X_val = np.load(npfiles[3])
y_val = np.load(npfiles[0])

# Display the first five elements of X_train
print("The first 5 elements of X_train are:\n", X_train[:5])  

# Display the first five elements of X_val
print("The first 5 elements of X_val are\n", X_val[:5]) 

# Display the first five elements of y_val
print("The first 5 elements of y_val are\n", y_val[:5]) 


# Create a scatter plot of the data. To change the markers to blue "x",
# we used the 'marker' and 'c' parameters
plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b') 

# Set the title
plt.title("The first dataset")
# Set the y-axis label
plt.ylabel('Throughput (mb/s)')
# Set the x-axis label
plt.xlabel('Latency (ms)')
# Set axis range
plt.axis([0, 30, 0, 30])
plt.show()


### 2.3 Gaussian distribution

# To perform anomaly detection, you will first need to fit a model to the data’s distribution.

# * Given a training set $\{x^{(1)}, ..., x^{(m)}\}$ you want to estimate the Gaussian distribution for each
# of the features $x_i$. 

# * Recall that the Gaussian distribution is given by

#    $$ p(x ; \mu,\sigma ^2) = \frac{1}{\sqrt{2 \pi \sigma ^2}}\exp^{ - \frac{(x - \mu)^2}{2 \sigma ^2} }$$

#    where $\mu$ is the mean and $\sigma^2$ is the variance.
#    
# * For each feature $i = 1\ldots n$, you need to find parameters $\mu_i$ and $\sigma_i^2$ 
# that fit the data in the $i$-th dimension $\{x_i^{(1)}, ..., x_i^{(m)}\}$ (the $i$-th dimension of each example).


def estimate_gaussian(X): 
    """
    Calculates mean and variance of all features 
    in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """

    var = np.var(X, axis=0)    
    mu = np.mean(X, axis=0)    

    return mu, var

# Estimate mean and variance of each feature
mu, var = estimate_gaussian(X_train)              

print("Mean of each feature:", mu)
print("Variance of each feature:", var)

### 2.4 Selecting the threshold, $\epsilon$


# ### Exercise 2
# Please complete the `select_threshold` function below to find the best threshold
#  to use for selecting outliers based on the results from the validation set (`p_val`) 
# and the ground truth (`y_val`). 

# * In the provided code `select_threshold`, there is already a loop that will try many different
#  values of $\varepsilon$ and select the best $\varepsilon$ based on the $F_1$ score. 

# * You need to implement code to calculate the F1 score from choosing `epsilon` as the threshold 
# and place the value in `F1`. 

#   * Recall that if an example $x$ has a low probability $p(x) < \varepsilon$, then it is classified
#  as an anomaly. 
        
#   * Then, you can compute precision and recall by: 
#    $$\begin{aligned}
#    prec&=&\frac{tp}{tp+fp}\\
#    rec&=&\frac{tp}{tp+fn},
#    \end{aligned}$$ where
#     * $tp$ is the number of true positives: the ground truth label says it’s an anomaly and our
#  algorithm correctly classified it as an anomaly.
#     * $fp$ is the number of false positives: the ground truth label says it’s not an anomaly,
#  but our algorithm incorrectly classified it as an anomaly.
#     * $fn$ is the number of false negatives: the ground truth label says it’s an anomaly, 
# but our algorithm incorrectly classified it as not being anomalous.

#   * The $F_1$ score is computed using precision ($prec$) and recall ($rec$) as follows:
#     $$F_1 = \frac{2\cdot prec \cdot rec}{prec + rec}$$ 

# **Implementation Note:** 
# In order to compute $tp$, $fp$ and $fn$, you may be able to use a vectorized implementation 
# rather than loop over all the examples.


# If you get stuck, you can check out the hints presented after the cell below to help you 
# with the implementation.
def select_threshold(y_val, p_val): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    ground_truth_indexes = np.where(y_val == 1)

    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
    
        # Find indexes of p_val larger than 
        anomaly_indexes = np.where(p_val < epsilon)

        # Truth positive samples 
        tp = np.sum(y_val[anomaly_indexes] == 1)

        # False positive samples
        fp = np.sum(y_val[anomaly_indexes] == 0)

        # False negative samples
        fn = 0
        for gt in ground_truth_indexes[0]:
            if gt not in anomaly_indexes[0]:
                fn += 1

        prec = tp / (tp +fp)
        rec = tp / (tp + fn)

        F1 = 2 * prec * rec / (prec + rec)

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1


# Intesect two arrays with numpy 



# p_val = multivariate_gaussian(X_val, mu, var)
p_val = np.array([0.04163207, 0.08190901, 0.04071578, 0.06190003, 0.07118676])
epsilon, F1 = select_threshold(y_val[:5], p_val)