
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import string

from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import confusion_matrix,roc_auc_score,log_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


features = ['label', 'message']
sms = pd.read_csv('C:\\Python Project\\Text Analysis\\spam\\SVM\\t.csv', header=None, names=features)
sms.head()

#to create a new column
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})

X=sms.message
y=sms.label

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
print(y_train)
print(X_test)

#intiliaze the vectozier
#fit: pre processing
vect=CountVectorizer()
vect.fit(X_train)


#change the string to numeric data
X_train_dtm=vect.transform(X_train)

X_test_dtm=vect.transform(X_test)


from sklearn import svm
model = svm.SVC(kernel='linear', C=1, gamma=1) 
    # there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(X_train_dtm, y_train)
model.score(X_train_dtm, y_train)

y_pred_class = model.predict(X_test_dtm)
#Predict Output
predicted= model.predict(X_test_dtm)

from sklearn import metrics
metrics.accuracy_score(y_test,y_pred_class)
metrics.confusion_matrix(y_test,y_pred_class)







from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns


# Create object
iris = load_iris()

# Create data
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 1. Instantiate
# Default kernel='rbf'
svm = SVC(kernel='rbf')

# 2. Fit
svm.fit(X_train, y_train)

# 3. Predict 
y_pred = svm.predict(X_test)

print(y_pred)


# Accuracy calculation
acc = accuracy_score(y_pred, y_test)
