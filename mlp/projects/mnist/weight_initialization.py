"""weight_initialization
~~~~~~~~~~~~~~~~~~~~~~~~

This program shows how weight initialization affects training.  In
particular, we'll plot out how the classification accuracies improve
using either large starting weights, whose standard deviation is 1, or
the default starting weights, whose standard deviation is 1 over the
square root of the number of input neurons.

"""

# Standard library
import json
import random
import sys

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

# My library
from mlp.ml_alg import mnist_loader
from mlp.ml_alg import network2


class Usage(Exception):
    """Catch exception."""

    def __init__(self, msg):
        """Initialize attriutes."""
        self.msg = msg


def main(argv=None):
    if argv == None:
        argv = sys.argv
    try:
        if len(argv) != 4:
            print "You need to input three parameters!"
            raise Usage(__doc__)
        filename, n, eta = argv[1:]
        n = int(n)
        eta = float(eta)
        data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
        run_network(filename, n, eta, data_url)
        make_plot(filename)
    except Usage, err:
        print >>sys.stderr, (sys.argv[0].split('/')[-1] + ": "
                             + str(err.msg))
        return 2


def run_network(filename, n, eta,data_url):
    """Train the network using both the default and the large starting
    weights.  Store the results in the file with name ``filename``,
    where they can later be used by ``make_plots``.

    """
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(
        data_url)
    net = network2.Network([784, n, 10], cost=network2.CrossEntropyCost)
    print "Train the network using the default starting weights."
    default_vc, default_va, default_tc, default_ta \
        = net.SGD(training_data, 30, 10, eta, lmbda=5.0,
                  evaluation_data=validation_data,
                  monitor_evaluation_accuracy=True)
    print "Train the network using the large starting weights."
    net.large_weight_initializer()
    large_vc, large_va, large_tc, large_ta \
        = net.SGD(training_data, 30, 10, eta, lmbda=5.0,
                  evaluation_data=validation_data,
                  monitor_evaluation_accuracy=True)
    f = open(filename, "w")
    json.dump({"default_weight_initialization":
               [default_vc, default_va, default_tc, default_ta],
               "large_weight_initialization":
               [large_vc, large_va, large_tc, large_ta]},
              f)
    f.close()


def make_plot(filename):
    """Load the results from the file ``filename``, and generate the
    corresponding plot.

    """
    f = open(filename, "r")
    results = json.load(f)
    f.close()
    default_vc, default_va, default_tc, default_ta = results[
        "default_weight_initialization"]
    large_vc, large_va, large_tc, large_ta = results[
        "large_weight_initialization"]
    # Convert raw classification numbers to percentages, for plotting
    default_va = [x/100.0 for x in default_va]
    large_va = [x/100.0 for x in large_va]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, 30, 1), large_va, color='#2A6EA6',
            label="Old approach to weight initialization")
    ax.plot(np.arange(0, 30, 1), default_va, color='#FFA933',
            label="New approach to weight initialization")
    ax.set_xlim([0, 30])
    ax.set_xlabel('Epoch')
    ax.set_ylim([85, 100])
    ax.set_title('Classification accuracy')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    sys.exit(main())
