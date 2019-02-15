
import matplotlib
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
#
df=pd.read_csv('filename_1.csv')
#
y = df['Creditability']

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV,cross_val_score

### build a classifier
clf = RandomForestClassifier()



def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# use a full grid over all parameters
param_grid = {"criterion": ["gini", "entropy"]}
y = df['Creditability']
x = df.drop(['Creditability'], axis=1)
#
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=2)
# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X_train, y_train)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
tr =RandomForestClassifier()
gsearch = GridSearchCV(tr,param_grid)
fits=gsearch.fit(X_train, y_train)
model = gsearch.best_params_
model
print(model)

dt_scores = cross_val_score(fits, X_train, y_train, cv = 5)
print("mean cross validation score: {}".format(np.mean(dt_scores)))
print("score without cv: {}".format(fits.score(X_train, y_train)))

predictions = gsearch.predict(X_test)

## confusion matrix ([TN,FN],[FP,TP])
confusion_matrix(y_test,predictions)

y_pred = gsearch.fit(X_train, y_train).predict_proba(X_test)[:, 1]

