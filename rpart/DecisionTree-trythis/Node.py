### CODE FOR Node.py ###
# Path: Node.py
# Description: This code is used to create a node for the decision tree created by DecisionTree.py.
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
