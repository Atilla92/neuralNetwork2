import numpy as np
import cv2
import sys, math
import matplotlib.pyplot as plt



#calculate frame rate:


samplingPoints = [76, 55, 124, 199, 39, 28, 66, 24, 99, 540]
velocities = [-0.13, -0.18, -0.08, -0.05, -0.25, -0.35, -0.15, -0.4, -0.1, -22]

fps = 200
dt = 1./fps


lStim = 0.45 #m  half length of stimulus
xStart = 50 

for i in range(np.size(samplingPoints)):
	velStim = velocities[i]
	samplingPoint = samplingPoints[i]
	# print velStim
	print samplingPoint
	dt = 1./fps 
	tEnd = 0
	tStart= xStart/velStim
	t = np.arange(tStart, tEnd, dt)
	print len(t)
	tframes=len(t)

	framesPerSample = tframes / samplingPoint
	print framesPerSample