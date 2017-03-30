"""
mnist_svm
~~~~~~~~~

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

#### Libraries
# Third-party libraries
from sklearn import svm
from time import time

# My libraries
from mlp.ml_alg import mnist_loader

def svm_baseline():
    print "Load data"
    data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    training_data, validation_data, test_data =  mnist_loader.load_data(data_url=data_url)
    # Train
    print "Train data"
    time_start = time()
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    # Test
    print "Test data"
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    train_test_time = time() - time_start
    print("Training and testing time: {0}".format(train_test_time))
    print "Baseline classifier using an SVM."
    print "%s of %s values are correct." % (num_correct, len(test_data[1]))

if __name__ == "__main__":
    svm_baseline()
