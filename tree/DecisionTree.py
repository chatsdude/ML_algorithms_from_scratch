import sys
import numpy as np
from collections import Counter
from utils.Node import Node

class DecisionTreeClassifier():

    def __init__(self,max_depth=100,min_samples=2,num_features=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
        self.num_features = num_features

    def fit(self, X, y):
        self.root = self._build_decision_tree(X, y)

    def _build_decision_tree(self, X, y,depth=0):
        #Get the samples and features from the data.
        n_samples,n_features = X.shape
        n_labels = len(np.unique(y))
        self.num_features = n_features if self.num_features is None else self.num_features

        #If data is empty, return straightaway.
        if n_labels==0 or n_samples==0:
            return

        #Stopping criteria.
        # 1. Check if max_depth is reached
        #2. Check if no. of samples in current split are less than min samples specified.
        #3. Check the distribution of samples after the split.
        if depth >= self.max_depth or n_labels==1 or n_samples<self.min_samples:
            leaf_value = self._find_most_common_label(y)
            return Node(value=leaf_value)

        #Shuffle the feature indexes
        feat_idxs = np.random.choice(n_features,self.num_features)

        #Find the best split column and value
        split_idx,split_value = self._find_best_split(X,y,feat_idxs)
        
        #Split based on that column
        left_idxs,right_idxs = self._split(X[:,split_idx],split_value)

        #Recursively build the tree
        left_branch = self._build_decision_tree(X[left_idxs,:],y[left_idxs],depth+1)
        right_branch = self._build_decision_tree(X[right_idxs,:],y[right_idxs],depth+1)

        return Node(split_idx,split_value,left_branch,right_branch)

    def _find_best_split(self,X,y,feature_idxs):
        best_gain = float("-inf")
        split_idx,split_value = None,None

        #Go through all the features and find which feature gives the maximum information gain.
        for feature in feature_idxs:
            X_column = X[:,feature]
            values = np.unique(X_column)

            for val in values:
                gain = self._info_gain(X_column,y,val)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_value = val

        return split_idx,split_value

    def _info_gain(self,X,y,value):

        #Entropy before splitting
        parent_entropy = self._calculate_entropy(y)

        #Split based on current question value.
        #For eg: is col_value greater than VALUE?
        left_idxs,right_idxs = self._split(X,value)

        #If either of indexes are empty, information gain is zero.
        #This split is not useful.
        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0

        #Calculate the weighted entropy of the left and right idxs.
        num_samples = len(y)
        num_left_samples,num_right_samples = len(left_idxs),len(right_idxs)

        entropy_left,entropy_right = self._calculate_entropy(y[left_idxs]),self._calculate_entropy(y[right_idxs])

        weighted_child_entropy = (num_left_samples/num_samples)*entropy_left + (num_right_samples/num_samples)*entropy_right

        #Apply the information gain formula
        informationGain = parent_entropy - weighted_child_entropy

        return informationGain

    def _split(self,X,threshold):
        #Split based on two conditions.
        #Flatten is used to get 1-D array from np.argwhere
        #Will return a list of indexes
        left_idxs = np.argwhere(X<=threshold).flatten()
        right_idxs = np.argwhere(X>threshold).flatten()
        return left_idxs,right_idxs

    def _find_most_common_label(self, y):
        count = Counter(y)
        return count.most_common(1)[0][0]

    def predict(self, X):
        #For prediction,just traverse the tree.
        if self.root is not None:
            return np.array([self._traverse_decision_tree(x,self.root) for x in X])
        else:
            print("Call the fit method before predict")
            return

    def _traverse_decision_tree(self,data,root):
        if not root:
            return
            
        #Decision tree traversal helper function
        if root._is_leaf_node():
            return root.value

        if data[root.feature] <= root.threshold:
            return self._traverse_decision_tree(data,root.left)

        if data[root.feature] > root.threshold:
            return self._traverse_decision_tree(data,root.right)

    def _calculate_entropy(self,y):
        #Calculate the gini entropy of the target variable.
        yFrequency =  np.bincount(y)
        probability = yFrequency/len(y)
        return -np.sum([p*np.log2(p) for p in probability if p>0])

    