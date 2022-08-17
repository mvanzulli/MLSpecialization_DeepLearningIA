# Train a logistic regression model using scikit-learn.

# Libraries
from sklearn.linear_model import LogisticRegression
import numpy as np

# Dataset 
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])


# Create a model 
lr_model = LogisticRegression()
# train it 
lr_model.fit(X, y)

# Plot score
print("Accuracy on training set:", lr_model.score(X, y))