# https://www.youtube.com/watch?v=0pP4EwWJgIU&index=5&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# import
from sklearn.linear_model import LogisticRegression

# instantiate 
logreg = LogisticRegression()

#fit
logreg.fit(X,y)

# predict
print logreg.predict(X)


y_pred = logreg.predict(X)

# Classification Accuracy
# Proportion of correct predictions
from sklearn import metrics

# Compare actual vs predicted
print metrics.accuracy_score(y, y_pred) # = "Training Accuracy"

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neigbors=5)
knn.fit(X,y)
y_pred = knn.predict(X)
print metrics.accuracy_score(y, y_pred)
