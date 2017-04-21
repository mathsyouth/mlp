import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler

from mlp.ml_alg.svm import SVM


def generate_blobs():
    X, y = make_blobs(n_samples=1000, centers=2,
                      n_features=2, random_state=1)
    # Scale the data to be centered at the origin with a unit
    # standard deviation.
    X = StandardScaler().fit_transform(X, y)
    # Change the class labels to be +1 and -1 instead of 0 and 1.
    y[y == 0] = -1
    return X, y


def generate_circles():
    X, y = make_circles(n_samples=500, noise=0.1,
                        factor=0.1, random_state=1)
    X = StandardScaler().fit_transform(X, y)
    y[y == 0] = -1
    return X, y


def test_blobs():
    X, y = generate_blobs()
    # Set model parameters and initial values.
    # Instantiate a model with a large  C  value (a hard margin).
    C = 1000.0
    initial_alphas = np.zeros(len(X))
    initial_b = 0.0
    initial_errors = np.zeros(len(X))
    kernel='linear_kernel'
    # Instantiate model
    model = SVM(X, y, C, initial_alphas, initial_b,
                initial_errors, kernel)
    # Initialize error cache
    model.errors = model.decision_function(X) - y
    # np.random.seed(0)
    model.train()
    plt.plot(range(len(model._obj)), model._obj)
    plt.show()
    fig, ax = plt.subplots()
    grid, ax = model.plot_decision_boundary(ax)
    plt.show()


def test_blobs_with_outlier():
    X, y = generate_blobs()
    # Add an outlier
    X_outlier = np.append(X, [0.1, 0.1])
    X = X_outlier.reshape(X.shape[0]+1, X.shape[1])
    y = np.append(y, 1)

    # Set model parameters and initial values
    C = 1.0
    initial_alphas = np.zeros(len(X))
    initial_b = 0.0
    initial_errors = np.zeros(len(X))
    kernel='linear_kernel'
    # Instantiate model
    model = SVM(X, y, C, initial_alphas, initial_b,
                initial_errors, kernel)
    # Initialize error cache
    model.errors = model.decision_function(X) - y

    model.train()
    fig, ax = plt.subplots()
    grid, ax = model.plot_decision_boundary(ax)
    plt.show()


def test_circles():
    X, y = generate_circles()
    # Set model parameters and initial values.
    # Instantiate a model with a large  C  value (a hard margin).
    C = 1000.0
    initial_alphas = np.zeros(len(X))
    initial_b = 0.0
    initial_errors = np.zeros(len(X))
    kernel='gaussian_kernel'
    # Instantiate model
    model = SVM(X, y, C, initial_alphas, initial_b,
                initial_errors, kernel)
    # Initialize error cache
    model.errors = model.decision_function(X) - y
    model.train()
    fig, ax = plt.subplots()
    grid, ax = model.plot_decision_boundary(ax)
    plt.show()


if __name__ == "__main__":
    TESTS = {
        "test_blobs": test_blobs,
        "test_blobs_with_outlier": test_blobs_with_outlier,
        "test_circles": test_circles
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--testcases', choices=TESTS, type=str,
                        default='test_blobs', help="list of testcases")
    args = vars(parser.parse_args())
    TESTS[args["testcases"]]()
