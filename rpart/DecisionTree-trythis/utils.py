import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    # Create an array of indices from 0 to the length of X
    idxs = np.arange(len(X))

    # Shuffle the indices
    np.random.shuffle(idxs)

    # Calculate the number of test samples
    n_test_samples = int(len(X) * test_size)

    # Split the indices into train and test sets
    test_idxs = idxs[:n_test_samples]
    train_idxs = idxs[n_test_samples:]

    # Use the indices to split X and y into train and test sets
    X_train, X_test = X[train_idxs], X[test_idxs]
    y_train, y_test = y[train_idxs], y[test_idxs]

    return X_train, X_test, y_train, y_test
