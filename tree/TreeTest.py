import sys
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTreeClassifier


iris=datasets.load_iris()
X,y=iris.data,iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1234)
tree=DecisionTreeClassifier()
tree.fit(X_train,y_train)
res=tree.predict(X_test)
print(res)
print(y_test)
acc=np.sum(res==y_test)/len(y_test)
print(acc)