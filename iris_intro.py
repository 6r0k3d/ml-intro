# https://www.youtube.com/watch?v=RlQuVL6-qe8&index=4&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A

from sklearn.datasets import load_iris


iris = load_iris()

print type(iris)


# Data = instances and associated attributes
# Each row is an instance
# Each column is a feature
X = iris.data

# store response vector
# Has same number of rows as training data
# This is the expected response
y = iris.target

# scikit-learn 4 step modeling pattern
# 1: Import the class you plan to use
from sklearn.neighbors import KNeighborsClassifier

# 2: Instantiate the estimator
#   "estimator" is sklearn term for model
#   specify tuning parameters (below, n_neighbors)


# knn is the name of the estimator. Other common names include clf
# for classifier

knn = KNeighborsClassifier(n_neighbors=1) 

# To see additional parameters, you can print out the estimator
print knn

# 3: Fit the model with data (train)
# Provide training data and response
# Occurs in place, don't need to assign results to a variable
knn.fit(X,y)

# 4: Make predictions for a new observation
print knn.predict([[3,5,4,2]])

# Can make multiple predictions at once
X_new = [[3,5,4,2],[5,4,3,2]]
print knn.predict(X_new)


# Tuning: Use different values
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)

print knn.predict(X_new)


