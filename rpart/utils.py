from operator import itemgetter
import numpy as np
from rpart.Node import Node
from scipy.stats import chi2_contingency
import pandas as pd
from graphviz import Digraph
import itertools as it
from functools import reduce


def entropy(x: np.ndarray):
    _, counts = np.unique(x, return_counts=True)
    probabilities = counts / len(x)
    return -np.sum(probabilities * np.log2(probabilities))


def gini(x: np.ndarray):
    _, counts = np.unique(x, return_counts=True)
    probabilities = counts / len(x)
    return 1 - np.sum(np.square(probabilities))


def chimerge(self, x: np.ndarray, y: np.ndarray):
    df = pd.DataFrame({"x": x, "y": y}).sort_values("x")
    intervals = [(row["x"], row["x"]) for _, row in df.iterrows()]

    while True:
        chi_values = []
        for i in range(len(intervals) - 1):
            interval_1, interval_2 = intervals[i], intervals[i + 1]
            mask_1 = (df["x"] >= interval_1[0]) & (df["x"] <= interval_1[1])
            mask_2 = (df["x"] >= interval_2[0]) & (df["x"] <= interval_2[1])
            crosstab = pd.crosstab(df[mask_1 | mask_2]["y"], mask_1[mask_1 | mask_2])
            chi_value = chi2_contingency(crosstab, correction=False)[0]
            chi_values.append(chi_value)

        min_chi_value = min(chi_values)
        if min_chi_value > self.chimerge_threshold or (
            self.chimerge_max_intervals and len(intervals) <= self.chimerge_max_intervals
        ):
            break

        min_chi_index = chi_values.index(min_chi_value)
        intervals[min_chi_index] = (
            intervals[min_chi_index][0],
            intervals[min_chi_index + 1][1],
        )
        del intervals[min_chi_index + 1]

    return intervals


def draw_tree(root_node: Node, feature_names: list[str]):
    dot = Digraph()
    node_counter = 0

    def draw_node(node, parent=None, edge_label=None):
        nonlocal node_counter, dot
        if node.is_leaf_node():
            label = f"Leaf: {node.value}"
        else:
            feature_name = feature_names[node.feature]
            if isinstance(node.threshold, (int, float)):
                threshold_str = f"{node.threshold:.2f}"
            else:
                threshold_str = str(node.threshold)

            label = f"Feature {feature_name} <= {threshold_str}"

        node_id = f"node{node_counter}"
        dot.node(node_id, label)

        if parent is not None:
            dot.edge(parent, node_id, label=edge_label)

        node_counter += 1

        if not node.is_leaf_node():
            draw_node(node.left, node_id, "True")
            draw_node(node.right, node_id, "False")

    draw_node(root_node)
    return dot


def reduceByKey(func, iterable):
    return (
        (k, reduce(func, (x[1] for x in xs)))
        for (k, xs) in it.groupby(sorted(iterable, key=itemgetter(0)), itemgetter(0))
    )
