"""Implementation for decision tree classifier with multiprocessing"""
import multiprocessing as mp


class DecisionTreeClassifier:
    def __init__(
        self,
        criterion="entropy",
        max_depth: int = None,
        min_samples_split: int = 2,
        n_workers: int = 5,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_workers = n_workers
        # decide on the tree data structure
        self.tree = None
        # decide on the split candidates
        pass

    def fit(self, data):
        """
        interface for model fitting
        """
        # partition the dataset
        self.data = data
        self._split_candidates = self._find_split_candidates(data)
        self._data_partitioned = self._partition_dataset(data)

        # parallel loop here
        with mp.Pool(self.n_workers) as pool:
            pool.map(self._fit_worker, self._data_partitioned)

    def predict(self, data=None):
        """

        if data is None, predict on the training data
        """
        pass

    def _fit_worker(self, partition):
        # for a partition of the dataset
        # get the metric for each split-candidate
        metrics = {
            split: self._get_metric(partition, split) for split in self._split_candidates
        }

        return metrics

    def _get_metric(self, partition, split):
        # get the metric for a split in a partition
        # use self.criterion here
        pass

    def _find_split_candidates(self, data):
        # find the split candidates
        pass

    def _partition_dataset(self, data):
        # partition the dataset
        pass
