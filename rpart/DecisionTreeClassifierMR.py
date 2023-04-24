from rpart.DecisionTreeClassifier import DecisionTreeClassifier
import numpy as np
import pandas as pd


class DecisionTreeClassifierMR(DecisionTreeClassifier):
    def __init__(
        self,
        max_depth: int = None,
        metric: str = "gini",
        split_method: str = None,
        chimerge_threshold: float = 0.05,
        chimerge_max_intervals: int = None,
        n_workers: int = 5,
    ):
        super().__init__(
            max_depth, metric, split_method, chimerge_threshold, chimerge_max_intervals
        )
        self.n_workers = n_workers

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            self.feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        else:
            self.feature_names = X.columns
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.feature_names = X.columns
        self._partitions = self._partition_data(X, y)
        self.root = self._grow_tree(X, y)

    def _partition_data(self, X):
        partitions = []
        partition_size = len(X) // self.n_workers

        for i in range(self.n_workers):
            if i == self.n_workers - 1:
                # For the last partition, include the remaining rows
                partition = X.iloc[i * partition_size :]
            else:
                partition = X.iloc[i * partition_size : (i + 1) * partition_size]

            partitions.append(partition)

        return partitions
