from DecisionTree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from utils.CustomException import CustomException
import traceback
from sklearn import datasets
from sklearn.model_selection import train_test_split

class RandomForestClassifier(DecisionTreeClassifier):

    def __init__(self,n_trees=20,max_depth=50,min_samples=2,num_features=None):
        self.n_trees = n_trees
        self.trees = list()
        super().__init__(max_depth,min_samples,num_features)

    def fit(self, X, y):
        try:
            valid_inputs = self.validate_training_inputs(X,y)

            if not valid_inputs:
                raise CustomException("Only numpy array and dataframe objects are supported as inputs.")

            #Convert X and y into dfs for sampling
            X,y = pd.DataFrame(X),pd.Series(y)

            X['Target'] = y

            #Create a bootstrapped dataset
            X = X.sample(n=len(X),replace=True)

            target = X['Target']
            X = X.drop('Target',axis=1)

            #Convert df back to np array
            X = X.to_numpy()
            target = target.to_numpy()

            #Create a decision tree for this data
            root = DecisionTreeClassifier._build_decision_tree(self,X,target)

            if root is not None:
                self.trees.append(root)
            else:
                raise CustomException("Random forest was not initialized properly. Try using different parameters.")


        except CustomException as e:
            traceback.print_exc()

    def validate_training_inputs(self,X,y):
        if (isinstance(X,np.ndarray) or isinstance(X,pd.DataFrame)) and (isinstance(y,np.ndarray) or isinstance(y,pd.Series)):
            return True

        else:
            return False

    def predict(self, X):
        prediction,predictions=[],[]

        #Pass a single datapoint and get prediction from all decision trees in random forest.
        #Select the most common result as the final prediction.
        for x in X:
            for root in self.trees:
                prediction.append(DecisionTreeClassifier._traverse_decision_tree(self,x,root))
            predictions.append(max(set(prediction),key=prediction.count))
            prediction.clear()

        return np.array(predictions)


if __name__ == '__main__':
    dataset=datasets.load_breast_cancer()
    X,y=dataset.data,dataset.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1234)
    rf = RandomForestClassifier()
    rf.fit(X_train,y_train)
    predictions = rf.predict(X_test)
    acc=np.sum(predictions==y_test)/len(y_test)
    print(acc)


