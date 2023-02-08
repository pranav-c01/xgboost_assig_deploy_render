# all necessary imports
# import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
# import sklearn.metrics as metrics
# import pickle

# Dataset build
train_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header = None)
test_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test' , skiprows = 1, header = None)
col_labels = ['age', 'workclass', 'fnlwgt', 'education','education_num',
'marital_status', 'occupation','relationship','race', 'sex','capital_gain',
'capital_loss','hours_per_week', 'native_country', 'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels


# Separating X and y datasets for both train and test data
x_train = train_set.drop(['wage_class'],axis=1)
y_train = train_set['wage_class']
x_test = test_set.drop(['wage_class'],axis=1)
y_test = test_set['wage_class']

# feature selection
cols_new = ['native_country','hours_per_week','capital_loss','capital_gain','sex','occupation','education_num','workclass']
x_train = x_train.loc[:,cols_new]
x_test = x_test.loc[:,cols_new]

# label encoding for categoricla cols
cat_cols = ['workclass','occupation','sex','native_country']
for i in cat_cols:
    le = LabelEncoder() # x label encoder
    x_train[i] = le.fit_transform(x_train[i])
    x_test[i] = le.transform(x_test[i])

le1 = LabelEncoder()  # y label encoder
y_train= le.fit_transform(y_train)
le2 = LabelEncoder() 
y_test= le2.fit_transform(y_test)

## Model build
bst = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, objective='binary:logistic')
# fit model
bst.fit(x_train, y_train)
# preds2 = bst.predict(x_test)

## Evaluating model -> (84.81051532461151% accuracy)
# print("\n",bst.score(x_test,y_test))

# pr = metrics.classification_report(y_test,preds2)
# print(pr)

## pickling models 
# with(open('label_enc.sav','wb')) as f:
#     pickle.dump(le,f)

# with(open('standard_scaler.sav','wb')) as f:
#     pickle.dump(scaler,f)

# with(open('model.pkl','wb')) as f:
#     pickle.dump(bst,f)
