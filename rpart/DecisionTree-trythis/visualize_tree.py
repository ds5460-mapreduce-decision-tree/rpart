### CODE FOR visualize_tree.py ###
# Path: visualize_tree.py
# Description: This code is used to visualize the decision tree created by DecisionTree.py and Node.py using graphviz.
from graphviz import Digraph

def export_graphviz(node, graph=None):
    if graph is None:
        graph = Digraph()
        graph.node(str(id(node)), f"Root\nFeature {node.feature}\nThreshold {node.threshold}")
    else:
        graph.node(str(id(node)), f"Feature {node.feature}\nThreshold {node.threshold}")

    if node.left:
        if node.left.is_leaf():
            graph.node(str(id(node.left)), f"Leaf\nClass {node.left.value}")
        else:
            export_graphviz(node.left, graph)
        graph.edge(str(id(node)), str(id(node.left)), label="<= Threshold")

    if node.right:
        if node.right.is_leaf():
            graph.node(str(id(node.right)), f"Leaf\nClass {node.right.value}")
        else:
            export_graphviz(node.right, graph)
        graph.edge(str(id(node)), str(id(node.right)), label="> Threshold")

    return graph
