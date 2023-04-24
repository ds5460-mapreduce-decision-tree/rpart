from math import log2
import numpy as np
import pandas as pd
import itertools


# Calculate Gini impurity for a given target variable (y)
def gini_impurity(y):
    # Check if input is a Pandas Series
    if isinstance(y, pd.Series):
        # Calculate the probability of each class
        p = y.value_counts() / y.shape[0]
        # Calculate Gini impurity
        gini = 1 - np.sum(p**2)
        return gini
    else:
        raise ("Object must be a Pandas Series.")


# Calculate entropy for a given target variable (y)
def entropy(y):
    # Check if input is a Pandas Series
    if isinstance(y, pd.Series):
        # Calculate the probability of each class
        a = y.value_counts() / y.shape[0]
        # Calculate entropy
        entropy = np.sum(-a * np.log2(a + 1e-9))
        return entropy
    else:
        raise ("Object must be a Pandas Series.")


# Calculate variance for a given target variable (y)
def variance(y):
    # Check if there is only one element
    if len(y) == 1:
        return 0
    else:
        # Calculate variance
        return y.var()


# Calculate information gain for a given split (mask) and target variable (y)
def information_gain(y, mask, func=entropy):
    # Calculate the number of samples in each group after the split
    a = sum(mask)
    b = mask.shape[0] - a

    # Check if either group is empty
    if a == 0 or b == 0:
        ig = 0
    else:
        # Calculate information gain based on variable type (numeric or categorical)
        if y.dtypes != "O":
            information_gain = (
                variance(y)
                - (a / (a + b) * variance(y[mask]))
                - (b / (a + b) * variance(y[-mask]))
            )
        else:
            information_gain = (
                func(y) - a / (a + b) * func(y[mask]) - b / (a + b) * func(y[-mask])
            )
    return information_gain


# Generate all possible combinations of categories for a categorical variable (a)
# Not sure we need this, need to replace the code in max_information_gain_split if dropped
def cat_options(a):
    a = a.unique()
    options = []
    for L in range(0, len(a) + 1):
        for subset in itertools.combinations(a, L):
            subset = list(subset)
            options.append(subset)
    return options[1:-1]


# Find the best split for a given feature (x) and target variable (y) based on information gain
def max_information_gain_split(x, y, func=entropy):
    split_value = []
    ig = []
    # Determine if the feature is numeric or categorical
    numeric_variable = True if x.dtypes != "O" else False
    # Create options according to variable type
    if numeric_variable:
        options = x.sort_values().unique()[1:]
    else:
        options = cat_options(x)
    # Calculate information gain for all values
    for val in options:
        mask = x < val if numeric_variable else x.isin(val)
        val_ig = information_gain(y, mask, func)
        # Append results
        ig.append(val_ig)
        split_value.append(val)
    # Check if there are more than 1 results if not, return False
    if len(ig) == 0:
        return (None, None, None, False)
    else:
        # Get results with highest information gain (IG)
        best_ig = max(ig)
        best_ig_index = ig.index(best_ig)
        best_split = split_value[best_ig_index]
        return (best_ig, best_split, numeric_variable, True)


# Find the best split for all features in the dataset (df) and target variable (y)
def best_split(df, y, func=entropy):
    best_ig = []
    best_split = []
    numeric_variable = []
    # Iterate over all features and find the best split for each
    for col in df.columns:
        ig, split, is_numeric, has_split = max_information_gain_split(df[col], y, func)
        best_ig.append(ig)
        best_split.append(split)
        numeric_variable.append(is_numeric)
    # Get results with highest IG
    best_ig = max(best_ig)
    best_ig_index = best_ig.index(best_ig)
    best_split = best_split[best_ig_index]
    numeric_variable = numeric_variable[best_ig_index]
    return (best_ig, best_split, numeric_variable)


# # Make a binary split of the data based on the variable, value, and type (numeric or categorical)
# def make_split(variable, value, data, is_numeric):
#     if is_numeric:
#         data_1 = data[data[variable] < value]
#         data_2 = data[(data[variable] < value) == False]
#     else:
#         data_1 = data[data[variable].isin(value)]
#         data_2 = data[(data[variable].isin(value)) == False]
#     return(data_1, data_2)


# Recursively build a decision tree based on the input data, target variable, and optional parameters
def make_tree(
    data,
    target,
    func=entropy,
    max_depth=None,
    min_samples_split=2,
    depth=0,
    target_factor=True,
):
    # placeholder for the decision tree
    return


# Make predictions for a test dataset using the decision tree
def predict(tree, test_data):
    predictions = []
    # Iterate over each row in the test dataset
    for _, row in test_data.iterrows():
        # Traverse the decision tree to obtain a prediction for each instance
        prediction = traverse_tree(tree, row)
        # Append the prediction to the list of predictions
        predictions.append(prediction)
        # Return the list of predictions
    return predictions


# Recursively traverse the decision tree to obtain a prediction for a single instance (row)
def traverse_tree(tree, row):
    # Base case: if the current node is a leaf node (prediction), return the prediction
    if not isinstance(tree, dict):
        return tree
    # Get the decision rule from the current node
    decision_rule = list(tree.keys())[0]
    feature, operator, value = decision_rule.split()
    # Determine the next subtree to traverse based on the decision rule
    if operator == "<=":
        if row[feature] <= float(value):
            subtree = tree[decision_rule]
        else:
            subtree = tree[f"{feature} > {value}"]
    else:
        if row[feature] > float(value):
            subtree = tree[decision_rule]
        else:
            subtree = tree[f"{feature} <= {value}"]
    # Recursively traverse the tree
    return traverse_tree(subtree, row)
