"""Run neural networks."""

import argparse

from time import time

from mlp.projects.mnist import mnist_loader
from mlp.ml_alg import network2

def main():
    data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(
        data_url=data_url)
    print("Training {0}".format("cross-entropy_nn"))
    print("=============================")
    estimator = network2.Network([784, 30, 10],
                                 cost=network2.CrossEntropyCost)
    time_start = time()
    estimator.large_weight_initializer()
    estimator.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data,
                  lmbda = 5.0, monitor_evaluation_accuracy=True,
                  monitor_training_accuracy=True)
    train_test_time = time() - time_start
    print("done")
    print("Training and testing time: {0}".format(train_test_time))


if __name__ == "__main__":
    main()
