import numpy as np
from tree.utils import Node
from collections import Counter

class DecisionTree():

    def __init__(self,max_depth=100,min_samples=2,num_features=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
        self.num_features = num_features
        pass

    def fit(self, X, y):
        self.num_features = X.shape[1] if self.num_features is None else self.num_features
        self.root = self._build_decision_tree(X, y)
        pass

    def _build_decision_tree(self, X, y,depth=0):
        n_samples,n_features = X.shape
        n_labels = len(np.unique(y))

        #Stopping criteria.Check if max_depth is reached

        if depth >= self.max_depth or n_labels==1 or n_samples<self.min_samples:
            leaf_value = self._find_most_common_label(y)
            return Node(value=leaf_value)

        pass

    def _find_most_common_label(self, y):
        count = Counter(y)
        return count.most_common(1)[0][0]

    def predict(self, X, y):
        pass

    def _calculate_entropy(self,y):
        yFrequency =  np.bincount(y)
        probability = yFrequency/len(y)
        return -np.sum([p*np.log2(p) for p in probability if p>0])

    