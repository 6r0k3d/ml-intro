from datetime import datetime

# https://www.youtube.com/watch?v=WHocRqT-KkU
# 4-step model for models
# Import
# Instantiate
# Fit
# Predict

# 4-step model for vectorization
# Import
# Instantiate
# Fit
# Transform (into document term matrix, or doc matrix)

import pandas as pd

path = 'sms.tsv'
sms = pd.read_table(path, header=None, names=['label', 'message'])

print(sms.shape)
print(sms.head(10))

# Check response distribution
print(sms.label.value_counts())

# Convert label to number
# Map: dictionary, key = old value, value = new value
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})

print(sms.head(10))

# define x and y for use with Count Vectorizer
# Vectorizer takes a 1D series and converts to 2D array
X = sms.message
y = sms.label_num
print(X.shape)
print(y.shape)

#split x and y intro training and testing sets
# Why split before vectorizing?
# Because reversing it means the hold out sets features will be learned
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
#vect.fit(X_train)
#X_train_dtm = vect.transform(X_train)
# Slightly faster way of combining above lines
X_train_dtm = vect.fit_transform(X_train)

X_test_dtm = vect.transform(X_test)
print(X_test_dtm.shape)

# Build and evaluate model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
t = datetime.now()
nb.fit(X_train_dtm,y_train)
run_time = datetime.now() - t
print(run_time.total_seconds() * 1000)

y_pred_class = nb.predict(X_test_dtm)

from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))
print(metrics.confusion_matrix(y_test, y_pred_class))

# Look at the data where mistakes were made
# in order to help determine why
# How might you be able to get it right
# false positive messages
# print(X_test[(y_pred_class == 1) & (y_test == 0)])
print(X_test[y_pred_class > y_test])

# false negative messages
print(X_test[y_pred_class < y_test])


# Naive Bayes does not produce well calibrated probabilities


# Model Comparison
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

t = datetime.now()
logreg.fit(X_train_dtm, y_train)
run_time = datetime.now() - t
print(run_time.total_seconds() * 1000)

y_pred_class = logreg.predict(X_test_dtm)
print(metrics.accuracy_score(y_test, y_pred_class))


X_train_tokens = vect.get_feature_names()
print(len(X_train_tokens))



