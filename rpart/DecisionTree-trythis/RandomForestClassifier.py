import numpy as np
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=100, min_samples_split=2, max_features="sqrt", criterion="entropy"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, criterion=self.criterion)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        y_pred_majority_votes, _ = mode(tree_preds, axis=0)
        return np.squeeze(y_pred_majority_votes)

    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)
