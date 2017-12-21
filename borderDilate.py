import os
import cv2
import input_data
import numpy as np
import tensorflow as tf


sess = tf.InteractiveSession()


### train a cnn model
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]),name = "weight")
b = tf.Variable(tf.zeros([10]), name = 'bias')

y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

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

sess.run(tf.global_variables_initializer())

### restoreModel
saver = tf.train.Saver()
saver.restore(sess,'./model/model.ckpt')

def maxFilter(res):
    img = np.array(res)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if j>1 and j<(img.shape[1]-2):
                res[i][j] = np.maximum(np.maximum(img[i][j+1],img[i][j-1]),img[i][j])

    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            if i>1 and i<(img.shape[0]-2):
                res[i][j] = np.maximum(np.maximum(img[i-1][j],img[i+1][j]),img[i][j])

    # img = np.array(res)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if j>1 and j<(img.shape[1]-2):
    #             res[i][j] = np.minimum(np.minimum(img[i][j+1],img[i][j-1]),img[i][j])
    #
    # for j in range(img.shape[1]):
    #     for i in range(img.shape[0]):
    #         if i>1 and i<(img.shape[0]-2):
    #             res[i][j] = np.minimum(np.minimum(img[i-1][j],img[i+1][j]),img[i][j])

file_path = ['realphoto.jpg','digits.png', 'handwriten.jpg',"1.jpg","5.jpg"]
target = cv2.imread(file_path[3])

cv2.imshow('original',target)
cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()        # Closes displayed windows

target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
if target.shape[0]>1000 or target.shape[1]>1000:
    target = cv2.resize(target,(600,600))

### border detect
mask = np.array(target)
mask = cv2.Canny(mask,50,150 );

# cv2.imshow('border',mask)
# cv2.waitKey(0)                 # Waits forever for user to press any key
# cv2.destroyAllWindows()        # Closes displayed windows

kernel_mor = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
mask = cv2.dilate(mask,kernel_mor)
mask = cv2.erode(mask,kernel_mor)

# cv2.imshow('border',mask)
# cv2.waitKey(0)                 # Waits forever for user to press any key
# cv2.destroyAllWindows()        # Closes displayed windows.

### construct digital img
def detectDigit(img, x, y):

    left = bottom = 1e5
    right = top = 0
    pixels = []
    pixels.append((x,y))

    left = bottom = 1e5
    right = top = 0
    while(len(pixels) > 0):
        # print len(pixels)
        coordinate = pixels.pop()
        x0 = coordinate[0]
        y0 = coordinate[1]
        img[x0][y0] = 254
        # print "(%d,%d)"%(x0,y0)

        if(x0 < left):
            left = x0
        if(x0 > right):
            right = x0
        if(y0 < bottom):
            bottom = y0
        if(y0 > top):
            top = y0

        if(x0<img.shape[0]-1):
            if(img[x0 + 1][y0] == 255):
                pixels.append( (x0 + 1,y0) )

        if(x0>0):
            if(img[x0 - 1][y0] == 255):
                pixels.append( (x0 - 1,y0) )

        if(y0<img.shape[1]-1):
            if(img[x0][y0 + 1] == 255):
                pixels.append( (x0,y0 + 1) )

        if(y0>0):
            if(img[x0][y0 - 1] == 255):
                pixels.append( (x0,y0 - 1) )

    if (top - bottom)*(right - left) > 100:
        part = img[left:right,bottom:top]
        predict(part)

def predict(digit):
    ## process images
    width = digit.shape[0]
    height = digit.shape[1]
    m = max(width,height) * 1.5
    w = (int)(m - width)/2
    h = (int)(m - height)/2
    digit = cv2.copyMakeBorder(digit,h,h,w,w,cv2.BORDER_CONSTANT,value=0)
    digit = cv2.resize(digit,(28,28))

    ## prediction
    digits = np.reshape(digit, (1,np.product(digit.shape)))
    lable = sess.run(y_conv, feed_dict={x: digits, y_: mnist.test.labels, keep_prob: 1.0})
    # print digits
    lable = np.array(lable)
    print np.where(lable[0] == np.max(lable))[0][0]

    cv2.imshow('digit',digit)
    cv2.waitKey(0)                 # Waits forever for user to press any key
    cv2.destroyAllWindows()

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i][j] > 100:
            mask[i][j] = 255
        else:
            mask[i][j] = 0

cv2.imshow('original',target)
cv2.imshow('border',mask)
cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()        # Closes displayed windows.

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if(mask[i][j] == 255):
            detectDigit(mask,i,j)

cv2.imshow('filtered_image',mask)
cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()        # Closes displayed windows

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

sess.close()
