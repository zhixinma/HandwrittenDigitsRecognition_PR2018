# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import tensorflow as tf
import input_data

### gpu config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.InteractiveSession(config=config)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
    # return a Variable
    # truncated_normal() outputs random values from a truncated normal distribution.
    # stddev: The standard deviation of the truncated normal distribution
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # return a Variable
    # constant(): create a constant
    # if shape not present, the shape of value is used
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # convolution
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # pooling
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]),name = "weight")
b = tf.Variable(tf.zeros([10]), name = 'bias')

sess.run(tf.initialize_all_variables())
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

## First layer
# so the value of W_conv1 is random
# 5*5 filter, depth = 1, output = 32
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# convert image into a 4D tensor (-1)?28*28*1
x_image = tf.reshape(x, [-1,28,28,1])

# h_conv1 is a 32*1 vector?
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 5*5 filter depth = 32? output = 64?
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.initialize_all_variables())


saver = tf.train.Saver()
saver.restore(sess,'./model/model.ckpt')

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

sess.close()
