import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import os


#Current directory
current_directory = os.path.join(os.path.dirname(__file__))
# path to data directory
data_directory = os.path.join(current_directory, "data")
#Load the dataset
data_set = input_data.read_data_sets(data_directory, one_hot=True)

number_of_inputs = 784
layer_1_nodes = 100
layer_2_nodes = 50
layer_3_nodes = 15

#input layer
with tf.variable_scope("input"):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

#True labels
with tf.variable_scope("y_true"):
    y_true = tf.placeholder(tf.float32, shape=(None, 10))

# Fully connected Layer 1
with tf.variable_scope("layer_1"):
    weights = tf.get_variable("weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# Fully connected Layer 2
with tf.variable_scope("layer_2"):
    weigths = tf.get_variable("weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases2', shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weigths) + biases)

#Fully Connected Layer 3
with tf.variable_scope("layer_3"):
    weigths = tf.get_variable("weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases3', shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weigths) + biases)

#Drop out layer
with tf.variable_scope("dropout_layer"):
    keep_prob = tf.placeholder(tf.float32)
    dropout_layer = tf.nn.dropout(layer_3_output, keep_prob)

# Output Layer - softmax
with tf.variable_scope("softmax_layer"):
    weights = tf.get_variable("weights4", shape=[layer_3_nodes, 10], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[10], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.softmax(tf.matmul(dropout_layer, weights) + biases)

#Calculate Loss
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true , logits=layer_3_output))
#Minimize loss using adam optimizer
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy_loss)
correct_prediction = tf.equal(tf.argmax(layer_3_output, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    for i in range(0, 20000):
        batch = data_set.train.next_batch(100)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={X: batch[0], y_true: batch[1], keep_prob:1.0})
            print('step %d, training accuracy %g' %(i, train_accuracy))
        train_step.run(feed_dict={X: batch[0], y_true: batch[1], keep_prob:0.5})

    print('test set accuracy is %g' %(accuracy.eval(feed_dict={X: data_set.test.images, y_true: data_set.test.labels, keep_prob:1.0})))
    print(" %s seconds " % (time.time() - start_time))
