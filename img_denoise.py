import cv2
import tensorflow as tf
import numpy as np
import input_data

#### prediction
### read image
def extractDigit(img, mask, step, threshold, x, y):
    threshold = 20
    ok = 0
    block = (x, y)
    background = []
    background.append(block)

    while(len(background) > 0):
        print len(background)
        block = background.pop()
        x = block[0]
        y = block[1]

        for i in range(step):
            for j in range(step):
                img[x+i][y+i] = 255

        if x < mask.shape[0]-2*step:
            check(img,x,y,step,threshold,ok)
            if ok == 1:
                background.append((x+step,y))

        if x >= step:
            check(img,x,y,step,threshold,ok)
            if ok == 1:
                background.append((x-step,y))

        if y < mask.shape[1]-2*step:
            check(img,x,y,step,threshold,ok)
            if ok == 1:
                background.append((x,y+step))

        if y >= step:
            check(img,x,y,step,threshold,ok)
            if ok == 1:
                background.append((x,y-step))

def check(img, x, y, step, threshold, ok):
    ok = 1
    for i in range(step):
        for j in range(step):
            if img[x+i][y+j] > threshold:
                ok = 0

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

file_path = ['realphoto.jpg','digits.png', 'handwriten.jpg',"1.jpg","5.jpg"]
target = cv2.imread(file_path[2])
target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
if target.shape[0]>1000 or target.shape[1]>1000:
    target = cv2.resize(target,(600,600))

Range = np.array(np.zeros(256))

### border detect
mask = np.array(target)
mask = cv2.Canny(mask,50,150 );
# mask = cv2.imdilate(mask,strel('disk',3))
# mask = cv2.imerode(mask,strel('disk',2))
kernel_mor = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
mask = cv2.dilate(mask,kernel_mor)
mask = cv2.erode(mask,kernel_mor)

# mean_kernel = np.array([[1/8,1/2,1/8],
#                         [1/2, 1 ,1/2],
#                         [1/8,1/2,1/8]])
#
# mask = cv2.filter2D(mask,-1,mean_kernel)


cv2.imshow('border',mask)
cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()        # Closes displayed windows


extractDigit(target, mask, 3, 50, 0, 200)

cv2.imshow('target',target)
cv2.imshow('3',mask)
cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()        # Closes displayed windows
