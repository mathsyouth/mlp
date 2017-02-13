"""
__main__.py

Run neural networks
"""

import argparse

from time import time

from ml_alg.basic_fnn import network
from ml_alg import mnist_loader

data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
training_data, validation_data, test_data =   mnist_loader.load_data_wrapper(data_url=data_url)

ESTIMATORS = {
    "shallow_nn": network.Network([784, 30, 10]),
}

parser = argparse.ArgumentParser()
parser.add_argument('--classifiers', nargs="+",
                    choices=ESTIMATORS, type=str,
                    default=['shallow_nn'], help="list of classifiers")
args = vars(parser.parse_args())

print(__doc__)

for name in sorted(args["classifiers"]):
    print("Training {0}".format(name))
    print("===================")
    estimator = ESTIMATORS[name]
    time_start = time()
    estimator.SGD(training_data, 20, 5, 2.0, test_data=test_data)
    train_test_time = time() - time_start
    print("done")
    print("Training and testing time: {0}".format(train_test_time))
