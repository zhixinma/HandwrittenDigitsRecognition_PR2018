import cv2
import numpy as np
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# for i in range(np.size(mnist.test.images)):
for i in range(10):
    digit = np.reshape(mnist.test.images[i],(28,28))
    lable = mnist.test.labels[i]
    for i in range(np.size(lable)):
        m = 0
        if lable[i] > m:
            m = lable[i]
            index = i
    print index

    cv2.imshow('testSet',digit)
    cv2.waitKey(0)                 # Waits forever for user to press any key
    cv2.destroyAllWindows()        # Closes displayed windows
