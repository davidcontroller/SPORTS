# -*- coding: utf-8 -*-
"""
@author: io
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from IPython.display import display
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.pipeline import _name_estimators
import warnings
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

#'FTHG','FTAG'
#'HTP','ATP','HTGD','ATGD'

loc = 'C:\\Users\\io\\Desktop\\ALGOTRADING\\sport\\'

data = pd.read_csv(loc + 'dataset_draws.csv')
data = data[data.MW > 1]
data = data.filter(['DRA','FTHG','FTAG','HTGD','ATGD','DiffPts','DiffFormPts'], axis=1)
data.head()

n_matches = data.shape[0]

n_features = data.shape[1] - 1

n_draws = len(data[data.DRA == 'H'])

draws_rate = (float(n_draws) / (n_matches)) * 100

print("Total number of matches: {}".format(n_matches))
print("Number of features: {}".format(n_features))
print("Number of draws {}".format(n_draws))
print("Draws rate: {:.2f}%".format(draws_rate))

X_all = data.drop(['DRA'],1)
y_all = data['DRA']

y_all = y_all.apply(lambda x: 1 if x=='H' else 0)
y_all.head()

cols = [['HTGD','ATGD','FTHG','FTAG']]

for col in cols:
    X_all[col] = scale(X_all[col])
    
def preprocess_features(X):
    output = pd.DataFrame(index = X.index)
    for col, col_data in X.iteritems():
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)                 
        output = output.join(col_data)  
    return output

X_all = preprocess_features(X_all)
print ("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))
    
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                    test_size = 0.8,
                                                    random_state = 42)

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    clf.fit(X_train, y_train)
  
    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    y_pred = clf.predict(features)
    return f1_score(target, y_pred, pos_label=1), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):  
    train_classifier(clf, X_train, y_train)   
    f1, acc = predict_labels(clf, X_train, y_train) 
    f1, acc = predict_labels(clf, X_test, y_test)
    print ("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))
    
clf_A = LogisticRegression(random_state = 42)
clf_B = SVC(random_state = 912, kernel='rbf')
clf_C = xgb.XGBClassifier(seed = 82)

train_predict(clf_A, X_train, y_train, X_test, y_test)
train_predict(clf_B, X_train, y_train, X_test, y_test)
train_predict(clf_C, X_train, y_train, X_test, y_test)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


# TODO: Create the parameters list you wish to tune
parameters = { 'learning_rate' : [0.1],
               'n_estimators' : [40],
               'max_depth': [3],
               'min_child_weight': [3],
               'gamma':[0.4],
               'subsample' : [0.8],
               'colsample_bytree' : [0.8],
               'scale_pos_weight' : [1],
               'reg_alpha':[1e-5]
             }  

clf = xgb.XGBClassifier(seed=2)
 
f1_scorer = make_scorer(f1_score,pos_label=1)

grid_obj = GridSearchCV(clf,
                        scoring=f1_scorer,
                        param_grid=parameters,
                        cv=5)


grid_obj = grid_obj.fit(X_train,y_train)

clf = grid_obj.best_estimator_
print (clf)

f1, acc = predict_labels(clf, X_train, y_train)
print ("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
    
f1, acc = predict_labels(clf, X_test, y_test)
print ("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))

model2 = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print (sum(predictions)/len(predictions))
      
submission = pd.DataFrame(predictions)

submission.to_csv(loc + "test_pred_draw2.csv", index=False)
X_test.to_csv(loc + "TEST2.csv", index=True)