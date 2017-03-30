"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import cPickle
import gzip
import os
from shutil import copyfileobj
from urllib2 import urlopen

# Third-party libraries
import numpy as np


def get_data_home(data_home=None):
    """Return the path of the data dir.

    This folder is used by some large dataset loaders to avoid downloading the data several times.

    By default the data dir is set to a folder named 'data' in the user home folder.

    Alternatively, it can be set by the 'DATA' environment variable or programmatically by giving an explicit folder path. The '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.
    """
    if data_home is None:
        data_home = os.environ.get('DATA',
                                os.path.join('~', 'data'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def fetch_data(data_url, data_home=None):
    """Fetch a data set from the data's URL

    If the file does not exist yet, it is downloaded from data's URL.

    Parameters
    ----------

    data_url :
        URL of the data set

    data_home : optional, default: None
        Specify another download and cache folder for the data sets. By default
        all the data is stored in '~/data' subfolders.

    Returns
    -------

    data_path :
        Local path for the data set
    """

    # Check if this data set has been already downloaded
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, 'mnist_lisa')
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    data_path = os.path.join(data_home, 'mnist.pkl.gz')
    # If the file does not exit, download it
    if not os.path.exists(data_path):
        try:
            data_url = urlopen(data_url)
        except HTTPError as e:
            if e.code == 404:
                e.msg = "Dataset 'MNIST' not found."
            raise
        # store the gz file
        try:
            with open(data_path, 'w+b') as gz_file:
                copyfileobj(data_url, gz_file)
        except:
            os.remove(data_path)
            raise
        data_url.close()
    return data_path


def load_data(data_url, data_home=None):
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    data_path = fetch_data(data_url, data_home)
    f = gzip.open(data_path, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper(data_url, data_home=None):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data(data_url, data_home)
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
