from tensorflow.examples.tutorials.mnist import input_data
import keras as ks
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labels = mnist.test.labels

'''
def initializeWeights(shape):
    initial_weights = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_weights)
'''

train_images = np.reshape(train_images, (-1, 28, 28, 1))
test_images = np.reshape(test_images, (-1, 28, 28, 1))

#conv2d = Conv2D(data_format='')

model = ks.models.Sequential()
model.add(Conv2D(32, (3,3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

adam = ks.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20, batch_size=100)
score = model.evaluate(test_images, test_labels, batch_size=100)

print(score)
