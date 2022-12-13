# Import libraries 
import numpy as np
import matplotlib.pyplot as plt
# from utils import *
# %matplotlib inline
 ## 1 - Implementing K-means -->

# The K-means algorithm is a method to automatically cluster similar -->
# data points together.  -->

# * Concretely, you are given a training set $\{x^{(1)}, ..., x^{(m)}\}$, and you want
# to group the data into a few cohesive “clusters”. 


# * K-means is an iterative procedure that
#      * Starts by guessing the initial centroids, and then 
#      * Refines this guess by 
#          * Repeatedly assigning examples to their closest centroids, and then 
#          * Recomputing the centroids based on the assignments.
         

# * In pseudocode, the K-means algorithm is as follows:

    # Initialize centroids
    # K is the number of clusters
    # centroids = kMeans_init_centroids(X, K)
    
    # for iter in range(iterations):
        # Cluster assignment step: 
        # Assign each data point to the closest centroid. 
        # idx[i] corresponds to the index of the centroid 
        # assigned to example i
        # idx = find_closest_centroids(X, centroids)

        # Move centroid step: 
        # Compute means based on centroid assignments
        # centroids = compute_means(X, idx, K)
    # ```


# * The inner-loop of the algorithm repeatedly carries out two steps: 
    # * (i) Assigning each training example $x^{(i)}$ to its closest centroid, and
    # * (ii) Recomputing the mean of each centroid using the points assigned to it. 
    
    
# * The $K$-means algorithm will always converge to some final set of means for the centroids. However,
#  that the converged solution may not always be ideal and depends on the initial setting of the centroids.
# * Therefore, in practice the K-means algorithm is usually run a few times with different random initializations. 
# * One way to choose between these different solutions from different random initializations is to choose the one with the lowest cost function value (distortion).



# Exercise 1 
#  
# In the “cluster assignment” phase of the K-means algorithm, the
# algorithm assigns every training example $x^{(i)}$ to its closest
# centroid, given the current positions of centroids. 

### Exercise 1

# Your task is to complete the code in `find_closest_centroids`. 
# * This function takes the data matrix `X` and the locations of all
# centroids inside `centroids` 
# * It should output a one-dimensional array `idx` (which has the same number 
# of elements as `X`) that holds the index  of the closest centroid (a value in 
# $\{1,...,K\}$, where $K$ is total number of centroids) to every training example .

# * Specifically, for every example $x^{(i)}$ we set
# $$c^{(i)} := j \quad \mathrm{that \; minimizes} \quad ||x^{(i)} - \mu_j||^2,$$
# where 
#  * $c^{(i)}$ is the index of the centroid that is closest to $x^{(i)}$ (corresponds 
# to `idx[i]` in the starter code), and 
#  * $\mu_j$ is the position (value) of the $j$’th centroid. (stored in `centroids` in the starter code)
 
def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): k centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """
    # Set K
    K = centroids.shape[0]
    
    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    # for each point compute the distances to all the centroids
    #  and keep the one who provides the min distance 
    for i in range(X.shape[0]):

        # select the point features
        x_i = X[i, :]
        # initilize vector of disntances
        dist_p2centroids = np.zeros(K, dtype=float)
        # compute for each centroid the distance from the point p
        
        for j in range(K):
            
            # select the centroid position 
            mu_j = centroids[j]
            # distance to the centroid j
            dist_ij = np.linalg.norm(np.subtract(x_i,mu_j))
            # save the distance into the vector
            dist_p2centroids[j] = dist_ij**2
        
        # find the minimum and fill idx 
        idx[i] = np.argmin(dist_p2centroids)

    return idx


# Exercise 2

# Please complete the `compute_centroids` below to recompute the value for each centroid

# * Specifically, for every centroid $\mu_k$ we set
# $$\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x^{(i)}$$ 

#     where 
#     * $C_k$ is the set of examples that are assigned to centroid $k$
#     * $|C_k|$ is the number of examples in the set $C_k$


# * Concretely, if two examples say $x^{(3)}$ and $x^{(5)}$ are assigned to centroid $k=2$,
# then you should update $\mu_2 = \frac{1}{2}(x^{(3)}+x^{(5)})$.

# If you get stuck, you can check out the hints presented after the cell below to help you
#  with the implementation.


# UNQ_C2
# GRADED FUNCTION: compute_centpods

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    num_points_cluster_centroids =  np.zeros((K, ))
    
    # Sum all the positions sorted by centroid index 
    for i in range(m): #For each point do 
        
        # select the closest centroid 
        index_c = idx[i]
        
        # sum the position in the row of the new centroids 
        c_index = centroids[index_c, :]
        point = X[i, :]
        centroids[index_c, :] = np.add(point, c_index) 
        num_points_cluster_centroids[index_c] += 1
        
    # Compute the mean 
    one_over_m = np.divide(1,num_points_cluster_centroids)
    for index_c in range(centroids.shape[0]):
        centroids[index_c, :] = np.multiply(centroids[index_c, :], one_over_m[index_c])
    
    return centroids, num_points_cluster_centroids