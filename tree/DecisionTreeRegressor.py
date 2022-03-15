import sys
import numpy as np
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
from base.DecisionTreeBase import DecisionTreeBase


class DecisionTreeRegressor(DecisionTreeBase):

    def __init__(self,max_depth=25,min_samples=2,num_features=None):
        super().__init__(max_depth=max_depth,min_samples=min_samples,num_features=num_features)


    def fit(self, X, y):
        super().fit_regressor(X, y)

    def predict(self, X):
        return super().predict(X)

    def _find_mse_value(self,y_true,y_pred):
        return np.mean((y_true-y_pred)**2)

    
if __name__ == '__main__':
    X,y = datasets.make_regression(n_samples=5000,n_features=7,random_state=42)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1234)
    dt = DecisionTreeRegressor()
    #print(f"Testing data: {X_test}")
    dt.fit(X_train,y_train)
    #print(f"Training completed with node: {dt.root}")
    res = dt.predict(X_test)
    print(f"Predictions: {res}")
    print(f"True values: {y_test}")
    mse = dt._find_mse_value(y_test,res)
    print(f"MSE: {mse}")
