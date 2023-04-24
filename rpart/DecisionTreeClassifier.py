import numpy as np
import pandas as pd
from collections import Counter
from rpart.utils import chimerge, entropy, gini, reduceByKey
from rpart.Node import Node
from rpart.utils import draw_tree
import multiprocessing as mp


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int = None,
        metric: str = "gini",
        split_method: str = None,
        chimerge_threshold: float = 0.05,
        chimerge_max_intervals: int = None,
        n_workers: int = 5,
    ):
        self.max_depth = max_depth
        self.metric = metric
        self.split_method = split_method
        self.chimerge_threshold = chimerge_threshold
        self.chimerge_max_intervals = chimerge_max_intervals
        self.n_workers = n_workers

    def fit(self, X, y):
        self._prepare_fit(X, y)
        self.root = self._grow_tree(X, y)

    def fit_mapreduce(self, X, y):
        self._prepare_fit(X, y)
        self.root = self._grow_tree_mapreduce(X, y)

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        return np.array([self._predict(x) for _, x in X.iterrows()])

    def draw(self):
        return draw_tree(self.root, self.feature_names)

    def _prepare_fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            self.feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        else:
            self.feature_names = X.columns
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.feature_names = X.columns

    def _grow_tree(self, X, y, depth=0):
        if self._should_stop(X, y, depth):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        else:
            feature, threshold = self._best_split(X, y)
            if feature is None:
                leaf_value = self._most_common_label(y)
                return Node(value=leaf_value)

            mask = X.iloc[:, feature] <= threshold
            left = self._grow_tree(X[mask], y[mask], depth + 1)
            right = self._grow_tree(X[~mask], y[~mask], depth + 1)
            return Node(feature, threshold, left, right)

    def _should_stop(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_labels == 1
            or n_samples < 2
            or np.all(X == X.iloc[0].values)
        ):
            return True
        return False

    def _grow_tree_mapreduce(self, X, y, depth=0, mask=None):
        if mask is not None:
            X = X[mask]
            y = y[mask]

        if self._should_stop(X, y, depth):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        else:
            partitions = self._partition_data(X, y)
            feature, threshold = self._best_split_mapreduce(partitions)
            if feature is None:
                leaf_value = self._most_common_label(y)
                return Node(value=leaf_value)

            mask = X.iloc[:, feature] <= threshold
            left = self._grow_tree_mapreduce(X, y, depth + 1, mask)
            right = self._grow_tree_mapreduce(X, y, depth + 1, ~mask)
            return Node(feature, threshold, left, right)

    def _best_split(self, X, y):
        best_split = None
        min_score = float("inf")
        feature_thresholds = self._get_thresholds(X, y)
        for item in feature_thresholds:
            feature, thresholds = item
            x = X.iloc[:, feature]
            for threshold in thresholds:
                mask = x <= threshold
                left, right = y[mask], y[~mask]
                score = self._split_score(left, right)
                if score < min_score:
                    best_split = (feature, threshold)
                    min_score = score
        return best_split

    def _best_split_mapreduce(self, partitions):
        with mp.Pool(self.n_workers) as pool:
            feature_threshold_scores = pool.map_async(self._best_split_mapper, partitions)

        feature_threshold_scores = [
            result for result in feature_threshold_scores.get() if result is not None
        ]
        feature, threshold = self._best_split_reducer(feature_threshold_scores)
        return feature, threshold

    def _best_split_mapper(self, partition):
        X, y = partition
        feature_thresholds = self._get_thresholds(X, y)
        feature_threshold_scores = []
        for item in feature_thresholds:
            feature, thresholds = item
            x = X.iloc[:, feature]
            for threshold in thresholds:
                mask = x <= threshold
                left, right = y[mask], y[~mask]
                score = self._split_score_mapreduce(left, right)
                feature_threshold_scores.append(((feature, threshold), score))
        return feature_threshold_scores

    def _best_split_reducer(self, feature_threshold_scores):
        sorted_scores = sorted(
            [
                (feature_threshold, score)
                for feature_threshold, score in reduceByKey(
                    lambda x, y: x + y, feature_threshold_scores
                )
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        feature, threshold = sorted_scores[0][0]
        return feature, threshold

    def _split_score(self, left, right):
        if self.metric == "gini":
            score_func = gini
        elif self.metric == "entropy":
            score_func = entropy
        else:
            raise ValueError(
                "Invalid metric. Supported metrics are 'gini' and 'entropy'."
            )

        p_left = len(left) / (len(left) + len(right))
        p_right = len(right) / (len(left) + len(right))
        score = p_left * score_func(left) + p_right * score_func(right)
        return score

    def _split_score_mapreduce(self, left, right):
        p_left = len(left) / (len(left) + len(right))
        p_right = len(right) / (len(left) + len(right))
        score = p_left * gini(left) + p_right * gini(right)

        return score

    def _get_thresholds(self, X, y):
        feature_thresholds = []
        for feature in range(self.n_features):
            x = X.iloc[:, feature]
            unique_values = np.unique(x)

            # If the feature is numeric, use midpoints; otherwise, use all unique values
            if np.issubdtype(x.dtype, np.number):
                if self.split_method == "chimerge":
                    thresholds = chimerge(
                        x, y, self.chimerge_threshold, self.chimerge_max_intervals
                    )
                else:
                    sorted_values = np.sort(unique_values)
                    thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2

            else:
                thresholds = unique_values

            feature_thresholds.append((feature, thresholds))
        return feature_thresholds

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def _predict(self, x):
        node = self.root
        while node.left:
            feature_name = self.feature_names[node.feature]
            if x[feature_name] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _partition_data(self, X, y):
        partitions = []
        partition_size = len(X) // self.n_workers

        for i in range(self.n_workers):
            if i == self.n_workers - 1:
                # For the last partition, include the remaining rows
                X_partition = X.iloc[i * partition_size :]
                y_partition = y.iloc[i * partition_size :]
            else:
                X_partition = X.iloc[i * partition_size : (i + 1) * partition_size]
                y_partition = y.iloc[i * partition_size : (i + 1) * partition_size]

            partitions.append((X_partition, y_partition))

        return partitions
