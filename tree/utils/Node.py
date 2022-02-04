
class Node(object):

    def __init__(self,feature,threshold,left,right,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def _is_leaf_node(self):
        return self.value is not None