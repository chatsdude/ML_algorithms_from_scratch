import numpy as np
import sys
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
import traceback
import pandas as pd 
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tree.DecisionTree import DecisionTreeClassifier

class AdaBoostClassifier(DecisionTreeClassifier):

    def __init__(self,n_learners=10,max_depth=1,min_samples=2,num_features=None):
        self.n_learners = n_learners
        self.performance_dict = dict()
        super().__init__(max_depth,min_samples,num_features)

    def fit(self, X,y):
        self.n_samples,self.n_features = X.shape
        #_y = np.where(y<=0,-1,1)

        #Create sample weights from initial sample size
        sample_weight = 1 / self.n_samples
        self.sample_weights = np.array([sample_weight]*self.n_samples)
        updated_sample_weights = self.sample_weights

        for _ in range(self.n_learners):

            X,y = self._create_dataset_using_sample_weights(X,y,updated_sample_weights)

            #Build a decision stump.
            node = super()._build_decision_tree(X,y)

            #Find the misclassified samples.
            misclassified_idxs = self._find_misclassified_samples(X,y,node)
            #print(f"Misclassified idxs: {misclassified_idxs}")

            #Calculate total error.
            total_error = self._calculate_misclassification_error(self.sample_weights,misclassified_idxs)
            #print(f"Total error is: {total_error}")
            
            #Calculate performance_of_stump
            performance_of_stump = self._calculate_performance_of_stump(total_error)

            #Store performance_of_stump.
            #This will be used while making predictions.
            self.performance_dict[node] = performance_of_stump

            #Update sample weights.
            updated_sample_weights = self._update_sample_weights(self.sample_weights,misclassified_idxs,performance_of_stump)

            #Create bins from sample weights.
            #Eg: 0-0.0512,0.0512-0.0678 etc. etc.
            #bins = self._create_bins_of_sample_weights(self.sample_weights)
            
            #Update original data using bins.
            #Use this data to create a new stump
            #Continue this process for as many learners specified.
            #X,y = self._update_data_using_sample_weights(X,y,bins)


    def _calculate_misclassification_error(self,sample_weights,misclassified_idxs):

        return np.sum(sample_weights[misclassified_idxs])

    def _find_misclassified_samples(self,X,y,node):
        self.misclassified_idxs = []
        idx = 0
        for x,y in zip(X,y):
            ans = super()._traverse_decision_tree(x,node)
            if ans!=y:
                self.misclassified_idxs.append(idx)
            
            idx+=1
        
        return self.misclassified_idxs


    def _calculate_performance_of_stump(self,total_error):
        EPS = 1e-10
        ratio = (1 - total_error) / (total_error + EPS)
        performance_of_stump = 0.5 * np.log(ratio)
        return performance_of_stump

    def _update_sample_weights(self,sample_weights,misclassified_idxs,performance_of_stump):
        mask = np.isin(np.arange(0,self.n_samples),misclassified_idxs)
        sample_weights[mask] *= np.exp(performance_of_stump)
        sample_weights[~mask] *= np.exp(-performance_of_stump)
        sample_weights/=np.sum(sample_weights)
        return sample_weights

    def _create_bins_of_sample_weights(self,sample_weights):
        bins =  np.cumsum(sample_weights)
        #Start bin range from 0
        if bins[0]!=0:
            bins = np.insert(bins,0,0) 
        return bins

    def _update_data_using_sample_weights(self,X,y,bins):
        new_X_train = []
        new_y_train = []

        for _ in range(len(X)):
            random_num = np.random.random_sample()
            idx = np.digitize(random_num,bins) - 1
            #print(idx)
            X_data = list(X[idx])
            y_data = y[idx]
            new_X_train.append(X_data)
            new_y_train.append(y_data)

        return np.array(new_X_train),np.array(new_y_train)

    def _create_dataset_using_sample_weights(self,X,y,sample_weights):
        #Convert X and y into dfs for sampling
        X,y = pd.DataFrame(X),pd.Series(y)

        X['Target'] = y

        #Create a bootstrapped dataset
        X = X.sample(n=len(X),replace=True,weights=sample_weights)

        target = X['Target']
        X = X.drop('Target',axis=1)

        #Convert df back to np array
        X = X.to_numpy()
        target = target.to_numpy()

        return X,target

    def predict(self, X):
        predictions,class_zero_score,class_one_score = [],0,0
        for x in X:
            for node in self.performance_dict:
                prediction = super()._traverse_decision_tree(x,node)
                if prediction == 0:
                    class_zero_score += self.performance_dict[node]
                elif prediction == 1:
                    class_one_score += self.performance_dict[node]

            if class_one_score > class_zero_score:
                predictions.append(1)
            else:
                predictions.append(0)
            class_zero_score,class_one_score = 0,0
                
        return np.array(predictions)


if __name__ == "__main__":
    dataset=datasets.load_breast_cancer()
    X,y=dataset.data,dataset.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
    ad = AdaBoostClassifier()
    ad.fit(X_train,y_train)
    res = ad.predict(X_test)
    print(res)
    print(y_test)
    acc=np.sum(res==y_test)/len(y_test)
    print(acc)