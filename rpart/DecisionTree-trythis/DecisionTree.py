### CODE FOR DecisionTree.py ###
# Path: DecisionTree.py
# Description: This code is used to create a decision tree using Spark.
import numpy as np
from pyspark import SparkContext
from Node import Node

class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _is_finished(self, depth, n_samples, n_class_labels):
        return (depth >= self.max_depth
                or n_class_labels == 1
                or n_samples < self.min_samples_split)

    def _entropy(self, y):
        # Optimized entropy calculation
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / y.size
        entropy = -np.sum(proportions * np.log2(proportions))
        return entropy
    
    def _gini(self, y):
        proportions = np.bincount(y) / len(y)
        gini = 1 - np.sum(np.square(proportions))
        return gini

    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        if self.criterion == "entropy":
            parent_loss = self._entropy(y)
        elif self.criterion == "gini":
            parent_loss = self._gini(y)
        else:
            raise ValueError("Invalid criterion specified. Supported values: 'entropy', 'gini'")

        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0:
            return 0

        if self.criterion == "entropy":
            child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        elif self.criterion == "gini":
            child_loss = (n_left / n) * self._gini(y[left_idx]) + (n_right / n) * self._gini(y[right_idx])

        return parent_loss - child_loss

    def _best_split(self, X, y, features):
        split = {'score': -1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth, n_samples, n_class_labels):
            most_common_label = np.argmax(np.bincount(y))
            return Node(value=most_common_label)

        # get best split
        rnd_feats = np.random.choice(n_features, n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def fit(self, X, y, n_partitions=2):
        # Get or create a SparkContext instance
        sc = SparkContext.getOrCreate()
        
        # Stack X and y horizontally and create an RDD
        data = np.column_stack((X, y))
        rdd = sc.parallelize(data, numSlices=n_partitions)

        # Build the tree using the RDD
        self.root = self._build_tree(rdd)
    
    def print_tree_stats(self, depth, n_samples, best_feat, best_thresh):
        spark = SparkSession.builder.getOrCreate()
        executor_id = spark._jsparkSession.sc().getExecutorId()
        print(f"Executor ID: {executor_id}, Depth: {depth}, Samples: {n_samples}, Best Feature: {best_feat}, Best Threshold: {best_thresh}")

    def _build_tree(self, rdd, depth=0):
        # Collect the data from the RDD, split it back into X and y
        X_y = rdd.collect()
        X, y = np.hsplit(X_y, [-1])
        y = y.ravel()
        
        # Get the shape of X and the number of class labels
        n_samples, n_features = X.shape
        n_class_labels = len(np.unique(y))

        # Check stopping criteria for building the tree
        if self._is_finished(depth, n_samples, n_class_labels):
            most_common_label = np.argmax(np.bincount(y))
            return Node(value=most_common_label)

        # Get the best feature and threshold to split the data
        rnd_feats = np.random.choice(n_features, n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # Recursively build the left and right children using distributed data
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        
        # Parallelize the left and right partitions using Spark and build the children nodes
        left_child = self._build_tree(sc.parallelize(X_y[left_idx]), depth + 1)
        right_child = self._build_tree(sc.parallelize(X_y[right_idx]), depth + 1)
        
        # Return a new Node with the best feature, threshold, and the left and right children
        return Node(best_feat, best_thresh, left_child, right_child)