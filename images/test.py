import cv2
a = cv2.imread('28_Sports_Fan_Sports_Fan_28_6.jpg')
import numpy as np

#b = np.transpose(a, (2, 0, 1))
cv2.imshow('aa', a)

a = np.array(a)
b = a.transpose(2, 0, 1)
print(b.shape)

cv2.imshow('bb', b)
cv2.waitKey(0)
