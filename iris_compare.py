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
print(logreg.predict(X))


y_pred = logreg.predict(X)

## Model Evaluation Procedures

# Classification Accuracy
# Proportion of correct predictions
from sklearn import metrics

# Compare actual vs predicted
print(metrics.accuracy_score(y, y_pred)) # = "Training Accuracy"

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))

# Problems with training and testing on same data 
# maximizing training accuracy rewards overly complex models that dont generalize well
# models will "overfit"

# Need to split data into training and testing sets
# Rule of Thumb: generally 20-40% split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=4)
print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

scores = []
k_range = range(1,26)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt

plt.plot(k_range,scores)
plt.xlabel('Value of K')
plt.ylabel('Testing Accuracy')
#plt.show()

# https://www.youtube.com/watch?v=3ZWuPVWq7p4&index=6&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A
# Regression metrics
# Mean Absolute Error: average of absolute value of errors / Easy to understand
# Mean Squared Error: average of square of errors / "punishes" larger error
# Root Mean Squared Error: interpretable in "y" units

# Error: difference between true and predicted values
from sklearn import metrics
# metrics.mean_absolute_error(actual, predicted)

# Feature Selection: Is a feature useful in making prediction?
# Does removing it make the model more accurate?


# Cross Validation
# use "folds" of data
# Split data set into equal chunks, use 1 chunk to test
# Use remaining to train
# Average the error
# 10 folds has experimentally been shown to produce most reliable estimate
# Use stratified sampling for classification problems
# eg if good/bad split is 80/20, each fold should be 80/20
# Use to tune model hyperparameters

from sklearn.model_selection import cross_val_score
# Cross_val_score handles splitting data into folds for us
k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

plt.plot(k_range,k_scores)
plt.xlabel('Values of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
#plt.show()
# Generally recommended to pick value of k which produces simplest model
# Based on graph above, best K values are 13-20
# Therefore, pick k = 20, since > k == simpler model

# Feature Selection Introduction: https://machinelearningmastery.com/an-introduction-to-feature-selection/

# Grid-Search CV
# Compre hyperparameter settings
# Once best parameters are identified, dont forget to train with all available data

# RandomizedSearchCV
# Reduce computational expense 

### https://www.youtube.com/watch?v=85dtiMz9tSo&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=9
## Evaluating a classification model
# Purpose of model evaluation
# How to choose between models? Model types / parameters / features
# Need a metric to quantify performance
## Procedures
# Train/Test on same data
# Train/Test split (faster / simple)
# K-fold cross-validation (better out of sample performance)
## Metrics
# Regression: Mean Abs. Error / Mean Squared Error / Root Mean Squared Error
# Classification: Accuracy

## Classification accuracy usage and limitations
# When comparing classification accuracy, need to compare with 'null accuracy'
# 'null accuracy': accuracy that could be achieved by always predicting the most frequent class
# Accuracy does not inform the underlying distribution (need to check null accuracy)
# Accuracy does not inform what "types" of errors the classifier is making


## Using a confusion matrix
# Table that descirbes the performance of a classification model
# metrics.confusion_matrix(TESTING DATA, PREDICTION DATA)
# ORDER MATTERS, sklearn expected test, prediction
## Binary Classification Confusion Matrix
# 2x2 box
# True Positive: correct positive prediction
# True Negative: correct negative prediction
# False Positive: incorrect positive pred (Type I error)
# False Negative: incorrect negative pred (Type II error)

### Confusion Matrix metrics
## Classification Accuracy: TP + TN / float(All values)
## Classification Error: "How often is the classifier incorrect?"
# Lower is better
# == "Misclassification Rate"
# FP + FN / float(All values)

## Sensitivity: When actual value is positive, how often is prediction correct?
# Higher is better
# "True positive rate" or "Recall"
# TP / float(TP + FN)

## Specificity (Higher is better)
# When actual value is negative, how often is the prediction correct?
# TN / float(TN + FP)

## False Positive Rate
# When the actual value is negative, how often is the prediction wrong?
# FP / float(TN + FP)

## Precision
# When a positive value is predicted, how often is the prediction correct?
# TP / float(TP + FP)

### Can't optimize for all metrics
# Which to pick?
# Depends on "objective"
# Spam Filter: Optimize for precision or specificty
# False Negatives (spam in inbox) better than False Positive (non-spam in spam filter)

# Fraud transaction detector
# Optimize for sensitivty
# False positives (normal flagged as fraud) more acceptable than False Negative (fraud not detected)


# classification threshold
# Adjusting threshold changes sensitivity and specifcity
# Increase threshold increases specificity
# Decrease threshold increases sensitivity

# ROC curve: see how sensitivity + specifcity are affected by various thresholds
# sklearn metrics.roc_curve(test, probability)


# Area Under the Curve (AUC)
