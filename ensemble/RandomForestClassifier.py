import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tree.base.RandomForestBase import RandomForestBase
from sklearn import datasets
from sklearn.model_selection import train_test_split

class RandomForestClassifier(RandomForestBase):

    def __init__(self,n_trees=20,max_depth=50,min_samples=2,num_features=None):
        super().__init__(n_trees=n_trees,max_depth=max_depth,min_samples=min_samples,num_features=num_features)

    def fit(self, X, y):
        super().fit_classifier(X, y)

    def predict(self, X):
        return super().predict_classifier(X)


if __name__ == '__main__':
    dataset=datasets.load_breast_cancer()
    X,y=dataset.data,dataset.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1234)
    rf = RandomForestClassifier()
    rf.fit(X_train,y_train)
    predictions = rf.predict(X_test)
    acc=np.sum(predictions==y_test)/len(y_test)
    print(acc)


