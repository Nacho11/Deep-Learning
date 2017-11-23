import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import time
import matplotlib.pyplot as plt

def initializeWeights(shape):
    initial_weights = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_weights)

def biasVariable(shape):
    initial_bias = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_bias)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def maxPool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def avgPool2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

#Current directory
current_directory = os.path.join(os.path.dirname(__file__))
# path to data directory
data_directory = os.path.join(current_directory, "data")
#Load the dataset
data_set = input_data.read_data_sets(data_directory, one_hot=True)

number_of_training_examples = data_set.train.images.shape[0]
number_of_pixels = data_set.train.images.shape[1]
number_of_classes = data_set.train.labels.shape[1]
input_x = tf.placeholder(tf.float32, [None, number_of_pixels])
x_image = tf.reshape(input_x, [-1, 28, 28, 1])

y_true = tf.placeholder(tf.float32, [None, number_of_classes])

'''First Convolutional Layer'''
W_conv1 = initializeWeights([5, 5, 1, 32])
b_conv1 = biasVariable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

'''First Pooling Layer'''
h_pool1 = avgPool2x2(h_conv1)

'''Second Convolutional Layer'''
W_conv2 = initializeWeights([5, 5, 32, 64])
b_conv2 = biasVariable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

'''Third Convolutional layer'''
W_conv3 = initializeWeights([5, 5, 64, 96])
b_conv3 = biasVariable([96])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

'''Second Pooling Layer'''
h_pool2 = avgPool2x2(h_conv3)

'''Densely Connected Layer'''
W_fc1 = initializeWeights([7 * 7 * 96, 1024])
b_fc1 = biasVariable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*96])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

'''Dropout'''
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

'''Readout Layer'''
W_fc2 = initializeWeights([1024, 10])
b_fc2 = biasVariable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

'''Training and Evaluate'''
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true , logits=y_conv))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy_loss)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    print(x_image)
    for i in range(20000):
        batch = data_set.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={input_x: batch[0], y_true: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' %(i, train_accuracy))
        train_step.run(feed_dict={input_x: batch[0], y_true: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={input_x: data_set.test.images, y_true:data_set.test.labels, keep_prob: 1.0}))
    print("--- %s seconds ---" % (time.time() - start_time))
