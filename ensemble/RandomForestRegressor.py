from sklearn import datasets
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tree.base.RandomForestBase import RandomForestBase
import numpy as np


class RandomForestRegressor(RandomForestBase):

    def __init__(self,n_trees=2,max_depth=5,min_samples=10,num_features=None):
        super().__init__(n_trees=n_trees,max_depth=max_depth,min_samples=min_samples,num_features=num_features)

    def fit(self, X, y):
        super().fit_regressor(X,y)

    def predict(self, X):
        return super().predict_regressor(X)




if __name__ == '__main__':
    X,y = datasets.make_regression(n_samples=5000,n_features=7,random_state=42)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1234)
    rf = RandomForestRegressor()
    #print(f"Testing data: {X_test}")
    rf.fit(X_train,y_train)
    #print(f"Training completed with node: {dt.root}")
    res = rf.predict(X_test)
    print(f"Predictions: {res}")
    print(f"True values: {y_test}")
    mse = np.mean((y_test - res)**2)
    print(f"MSE: {mse}")