import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time


# Step 1: read the data
# using TF Learn's built-in function to load MNIST data the folder data/mnist
MNIST = input_data.read_data_sets("data/mnist", one_hot=True)

# Step 2: define parameters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 25

# Step 3: create placeholders for features and labels
# Each image in the MNIST data is of shape 28*28 = 784.
# Therefore, each image is represented with a 1 * 784 tensor.
# There are ten classes for all images, corresponding to 0 ~ 9
# Each label is one hot vector
X = tf.placeholder(tf.float32, [batch_size, 784], name="images")
Y = tf.placeholder(tf.float32, [batch_size, 10], name="labels")

# Step 4: create weights and bias
# W is initialized to random variables with mean of 0 and stddev of 0.01
# b is initialized to 0
# Shape of W is dependent on the dimensions of X and Y
# Shape of b is dependent on Y
W = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1, 10]), name="bias")

# Step 5: Y from X and W, b
# The model returns probability distribution of possible label of the image
# through the softmax layer.
# A batch_size * 10 tensor that represents the possibility of the digits
logits = tf.matmul(X, W) + b

# Step 6: define the loss function
# Use the  softmax cross entropy with logits as the loss function
# Compute mean cross entropy over examples in the batch.
# Softmax is applied internally.
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits, name="loss")
loss = tf.reduce_mean(entropy)

# Step 7: define training op
# Use gradient descent with learning rate of learning_rate to minize the loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # To visualize using TensorBoard
    writer = tf.summary.FileWriter("graphs/logistic_regression", sess.graph)

    start_time = time.time()
    sess.run(init)
    n_batches =  int(MNIST.train.num_examples/batch_size)
    for i in range(n_epochs):
        total_loss = 0
        for _ in range(n_batches):
            X_batch, Y_batch = MNIST.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss],
                                     feed_dict={X: X_batch, Y: Y_batch})
            total_loss += loss_batch
        print "Averagy loss for the epoch {0} is: {1}".format(i,
                                                          total_loss/n_batches)
    print "Total training time is {} seconds.".format(time.time() - start_time)
    print "Optimization is Finished."

    # Test the model
    n_batches = int(MNIST.test.num_examples/batch_size)
    total_num_correct_preds = 0
    for _ in range(n_batches):
        X_batch, Y_batch = MNIST.test.next_batch(batch_size)
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits],
            feed_dict={X: X_batch, Y: Y_batch})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        # Similar to numpy.count_nonzero(boolarray)
        num_correct_preds = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_num_correct_preds += sess.run(num_correct_preds)
    print "The accuracy of predicting test images is:  {0}".format(
        total_num_correct_preds/MNIST.test.num_examples)

    writer.close()
