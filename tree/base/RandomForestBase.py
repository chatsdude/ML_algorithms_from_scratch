import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from DecisionTreeBase import DecisionTreeBase
from utils.CustomException import CustomException
import traceback
import numpy as np
import pandas as pd

class RandomForestBase(DecisionTreeBase):

    def __init__(self,n_trees,max_depth,min_samples,num_features=None):
        self.n_trees = n_trees
        self.trees = list()
        super(RandomForestBase, self).__init__(max_depth=max_depth,min_samples=min_samples,num_features=num_features)
        

    def fit_classifier(self,X,y):
        try:
            valid_inputs = self.validate_training_inputs(X,y)

            if not valid_inputs:
                raise CustomException("Only numpy array and dataframe objects are supported as inputs.")

            
            for _ in range(self.n_trees):
            
                X,y = self._create_bootstrap_data(X,y)

                #Create a decision tree for this data
                root = super()._build_decision_tree_classifier(X,y)

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

    def predict_classifier(self, X):
        prediction,predictions=[],[]

        #Pass a single datapoint and get prediction from all decision trees in random forest.
        #Select the most common result as the final prediction.
        for x in X:
            for root in self.trees:
                prediction.append(DecisionTreeBase._traverse_decision_tree(self,x,root))
            predictions.append(max(set(prediction),key=prediction.count))
            prediction.clear()

        return np.array(predictions)

    def fit_regressor(self,X,y):
        try:
            valid_inputs = self.validate_training_inputs(X,y)

            if not valid_inputs:
                raise CustomException("Only numpy array and dataframe objects are supported as inputs.")

            
            for _ in range(self.n_trees):
            
                X,y = self._create_bootstrap_data(X,y)

                #Create a decision tree for this data
                root = super()._build_decision_tree_for_regression(X,y)

                if root is not None:
                    self.trees.append(root)
                else:
                    raise CustomException("Random forest was not initialized properly. Try using different parameters.")

        except CustomException as e:
            traceback.print_exc()

    @staticmethod
    def _create_bootstrap_data(X,y):
        #Convert X and y into dfs for sampling
        X,y = pd.DataFrame(X),pd.Series(y)

        X['Target'] = y

        #Create a bootstrapped dataset
        X = X.sample(n=len(X),replace=True)

        y = X['Target']
        X = X.drop('Target',axis=1)

        #Convert df back to np array
        X = X.to_numpy()
        y = y.to_numpy()

        return X,y

    
    def predict_regressor(self,X):
        prediction,predictions=[],[]

        #Pass a single datapoint and get prediction from all decision trees in random forest.
        #Select the average of the result as the final prediction.
        for x in X:
            for root in self.trees:
                output = super()._traverse_decision_tree(x,root)
                #print(output,type(output))
                if output!= "Fuck my life":
                    prediction.append(output)
        
            predictions.append(np.mean(prediction))
            prediction.clear()
        
        return np.array(predictions)

