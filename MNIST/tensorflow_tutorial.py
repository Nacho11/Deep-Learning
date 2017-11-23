import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
import os


#Current directory
current_directory = os.path.join(os.path.dirname(__file__))
# path to data directory
data_directory = os.path.join(current_directory, "data")
#Load the dataset
data_set = input_data.read_data_sets(data_directory, one_hot=True)

# get number of training examples and number of pixels
number_of_training_examples = data_set.train.images.shape[0]
number_of_pixels = data_set.train.images.shape[1]

#print(data_set.train.labels[:][0])
number_of_classes = data_set.train.labels.shape[1]


input_x = tf.placeholder(tf.float32, [None, number_of_pixels]) # None -- The dimension of input can be any length
#initialize W matrix and b vector
W = tf.Variable(tf.zeros([number_of_pixels, number_of_classes]))
b = tf.Variable(tf.zeros([number_of_classes]))

y_predicted = tf.nn.softmax(tf.matmul(input_x, W) + b)

#Training
y_true = tf.placeholder(tf.float32, [None, number_of_classes])
# cross_entropy - 1st way
#cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_predicted), reduction_indices[1]))
# cross_entropy - 2nd way
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predicted, labels=y_true))
train_step = tf.train.AdamOptimizer(0.007).minimize(cross_entropy_loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = data_set.train.next_batch(100)
    sess.run(train_step, feed_dict={input_x: batch_xs, y_true: batch_ys})

correct_prediction = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={input_x: data_set.test.images, y_true: data_set.test.labels}))
