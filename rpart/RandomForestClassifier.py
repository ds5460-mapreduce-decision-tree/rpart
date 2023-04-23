from collections import Counter
import multiprocessing as mp

import numpy as np
import pandas as pd
from rpart.DecisionTreeClassifier import DecisionTreeClassifier


def _fit_single_tree(args):
    X, y, params = args

    tree = DecisionTreeClassifier(
        max_depth=params["max_depth"],
        metric=params["metric"],
        split_method=params["split_method"],
        chimerge_threshold=params["chimerge_threshold"],
        chimerge_max_intervals=params["chimerge_max_intervals"],
    )

    if params["bootstrap"]:
        X_sample, y_sample = params["_bootstrap_sample"](X, y)
    else:
        X_sample, y_sample = X.copy(), y.copy()

    X_sample = params["_sample_features"](X_sample)
    tree.fit(X_sample, y_sample)
    return tree


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        metric="gini",
        n_workers=5,
        split_method=None,
        chimerge_threshold=0.05,
        chimerge_max_intervals=None,
        max_features="sqrt",
        bootstrap=True,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.metric = metric
        self.n_workers = n_workers
        self.split_method = split_method
        self.chimerge_threshold = chimerge_threshold
        self.chimerge_max_intervals = chimerge_max_intervals
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []

    def _sample_features(self, X):
        if self.max_features == "sqrt":
            n_features = int(np.sqrt(X.shape[1]))
        elif self.max_features == "log2":
            n_features = int(np.log2(X.shape[1]))
        elif isinstance(self.max_features, int):
            n_features = self.max_features
        else:
            raise ValueError(
                "Invalid max_features. Supported values are 'sqrt', 'log2', or an integer."
            )

        return X.sample(n=n_features, axis=1)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X.iloc[indices], y.iloc[indices]

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        params = {
            "max_depth": self.max_depth,
            "metric": self.metric,
            "split_method": self.split_method,
            "chimerge_threshold": self.chimerge_threshold,
            "chimerge_max_intervals": self.chimerge_max_intervals,
            "bootstrap": self.bootstrap,
            "_bootstrap_sample": self._bootstrap_sample,
            "_sample_features": self._sample_features,
        }

        with mp.Pool(processes=self.n_workers) as pool:
            results = pool.map_async(
                _fit_single_tree, [(X, y, params) for _ in range(self.n_estimators)]
            )
            self.trees = results.get()

    def _fit_single_tree(self, X, y):
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            metric=self.metric,
            split_method=self.split_method,
            chimerge_threshold=self.chimerge_threshold,
            chimerge_max_intervals=self.chimerge_max_intervals,
        )

        if self.bootstrap:
            X_sample, y_sample = self._bootstrap_sample(X, y)
        else:
            X_sample, y_sample = X.copy(), y.copy()

        X_sample = self._sample_features(X_sample)
        tree.fit(X_sample, y_sample)
        return tree

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        predictions = np.empty((self.n_estimators, X.shape[0]), dtype=object)

        for i, tree in enumerate(self.trees):
            predictions[i] = tree.predict(X)

        most_common = np.array(
            [Counter(predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        )
        return (
            most_common.astype(int)
            if np.issubdtype(most_common.dtype, np.number)
            else most_common
        )

    def predict_proba(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(
            lambda x: np.bincount(x) / x.size, axis=0, arr=tree_preds
        )
