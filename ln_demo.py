import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import r2_score
import sklearn.metrics as sk
from sklearn.metrics import confusion_matrix
from sklearn.metrics import explained_variance_score
warnings.filterwarnings('ignore')

# Import the numpy and pandas package
import numpy as np
import pandas as pd

# Read the given CSV file, and view some sample records
advertising = pd.read_csv("pro_1.csv")
advertising

X = advertising['access']
y = advertising['participants']
print("---------------------Training the model---------------------------")
X_train, X_test, y_train, y_test =  train_test_split(X,y, train_size = 0.7, test_size = 0.3, random_state = 100)
print("The X_train data looks like this after splitting", X_train)
print("The Y_train data looks like this after splitting", y_train)
X_train_sm = sm.add_constant(X_train)
lr =  sm.OLS(y_train, X_train_sm).fit()
print(lr.params)
y_train_pred = lr.predict(X_train_sm)
print("---------------------Error Terms-----------------------------------")
res = (y_train - y_train_pred)
fig = plt.figure()
sns.distplot(res, bins = 15)
plt.title('Error Terms', fontsize = 15)
plt.xlabel('y_train - y_train_pred', fontsize = 15)
plt.show()

X_test_sm = sm.add_constant(X_test)

# Predicting the y values corresponding to X_test_sm
y_test_pred = lr.predict(X_test_sm)
print("Predicted  Values")
print(y_test_pred)

r_squared = r2_score(y_test, y_test_pred, multioutput = 'variance_weighted')
print (r_squared)
print("Accuracy score =", round(sk.explained_variance_score(y_test, y_test_pred), 2))
print("Error/Loss Score=", round(sk.median_absolute_error(y_test, y_test_pred), 2))
print(explained_variance_score(y_test, y_test_pred, multioutput='raw_values'))
plt.scatter(X_test, y_test)
plt.plot(X_test, y_test_pred, 'r')
plt.show()
