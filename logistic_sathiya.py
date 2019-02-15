import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#

import sys
import types
import pandas as pd

raw_data = pd.read_csv('C:/Python Project/Logistic Method with fiunction call/German_credit.csv')
raw_data.shape
raw_data.describe()
raw_data.info()

raw_data.hist(figsize=(20,15))
plt.show()
raw_data.nunique()

print(raw_data.groupby('Creditability').size())


#COnvert foreign to float
print ("Original: \n") 
print (raw_data.Foreign_Worker.value_counts(), "\n")
raw_data['Foreign_Worker'] = pd.get_dummies(raw_data.Foreign_Worker,dtype=float,drop_first=True)

#COnvert govt to float
print ("Original: \n") 
print (raw_data.Occupation.value_counts(), "\n")
raw_data['Occupation'] = pd.get_dummies(raw_data.Occupation,dtype=float,drop_first=True)
raw_data.to_csv('gjgjgh.csv')

from missing_value import Missing_value
raw_data=Missing_value(raw_data)
#print(b)

aa=raw_data.isnull().sum()

total_len = len(raw_data['Creditability'])
percentage_labels = (raw_data['Creditability'].value_counts()/total_len)*100
percentage_labels

cor=raw_data.corr()
cor['Creditability'].sort_values(ascending=False)

# Spliting Target Variable
predictor= raw_data.iloc[:, raw_data.columns != 'Creditability']
target= raw_data.iloc[:, raw_data.columns == 'Creditability']

# split the dataset into train & test
from sklearn.cross_validation import train_test_split
x_train,x_test, y_train, y_test = train_test_split(predictor, target, test_size = 0.30, random_state=0)
print("x_train ",x_train.shape)
print("x_test ",x_test.shape)
print("y_train ",y_train.shape)
print("y_test ",y_test.shape)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
Y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
acc_log


from sklearn.metrics import classification_report

print(classification_report(y_test, Y_pred))

# Checking the accuracy with test data
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,Y_pred))

#Predicting proba
from sklearn.metrics import roc_curve,confusion_matrix
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, Y_pred)
confusion_matrix(y_test, Y_pred)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()








