"""Run neural networks."""

import argparse

from time import time

from mlp.projects.mnist import mnist_loader
from mlp.ml_alg import network


def main():
    data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(
        data_url=data_url)
    print("Training {0}".format("shallow_nn"))
    print("=============================")
    estimator = network.Network([784, 30, 10])
    time_start = time()
    estimator.SGD(training_data, 20, 10, 2.0, test_data=test_data)
    train_test_time = time() - time_start
    print("done")
    print("Training and testing time: {0}".format(train_test_time))


if __name__ == "__main__":
    main()
