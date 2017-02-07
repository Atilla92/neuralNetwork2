
import numpy as np

import cv2
import matplotlib.pylab as plt


im = np.zeros([30, 30])
# im2 = np.zeros([30,30])
# # Add target
im[10:25, 10:25] = 255.0
# im2[9:26,9>26] = 255.0

plt.imshow(im, cmap=plt.cm.Greys, interpolation='nearest')
plt.show()


# # Add noise
im += 50 * np.random.rand(im.shape[0], im.shape[1])

plt.imshow(im, cmap=plt.cm.Greys, interpolation='nearest')
plt.show()

kernel1 = np.array([[ 3, 10,  3],
                   [ 0,  0,  0],
                   [-3, -10, -3]])

kernel1 = np.array([[ 1, 2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]])
kernel1 = kernel1 / np.linalg.norm(kernel1)
print kernel1

kernel2 = kernel1.T
print kernel2

plt.subplot(121)
im_edge1 = cv2.filter2D(im, cv2.CV_64F, kernel1)
plt.imshow(im_edge1, cmap=plt.cm.gray, interpolation='nearest')
plt.show()

cv2.waitKey(0)
cap = cv2.VideoCapture('/home/atilla/Documents/Test/Neural_Network2/Stimulus/Data/WCS_fps_30.0_v_-22_l_0.45_x_70_Sc_(0, 0, 0)_Res_40_PixRes_7.avi')

while(cap.isOpened()):
    ret, frame = cap.read()
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray= cv2.resize(gray1, (500,500)) 
    im_edge1 = cv2.filter2D(gray, -1 , kernel1)
	#im_edge2 = cv2.filter2D(gray, cv2.CV_64F, kernel2)

    cv2.imshow('frame',im_edge1)#, cmap=plt.cm.Greys, interpolation='nearest')
    plt.imshow(im_edge1, cmap=plt.cm.RdGy, interpolation='nearest')
    plt.show()
    if cv2.waitKey(100) & 0xFF == ord('q'):
          break

cap.release()
cv2.destroyAllWindows()