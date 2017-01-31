import numpy as np
import cv2
import sys, math
import matplotlib.pyplot as plt
from sys import path

#path.append('/home/atilla/Documents/Test/Neural_Network/Stimulus/')

#import naturalBackground.py

velStim = -22 #m/s velocity of stimulus
	#velStim = -0.08  #m/s velocity of stimulus
	
lStim = 0.45 #m  half length of stimulus
xStart = 50 #m start approach distance away from specimen 

# Set locust parameters
xFov = 180 # degrees, fov locust azimuth
yFov = 180 # degrees, fov locust elevation
angPix = 0.2 # deg/pixel interommatidial angle specimen

scaleCon = (0, 0, 0)
backgroundColor = 255  # 0 black, 255 white 
scaleRes = 10  # resolution scale, to downsample later
fps = 100 # Frame Resolution
numFrames = 10 # Number of repetition frames at end and beginning 
#Starting posit


def getParamters(velStim, xStart, fps, lStim, angPix ):
	dt = 1./fps 
	tEnd = 0
	tStart= float(xStart/velStim)
	t = np.arange(tStart, tEnd, dt)
	print t
	xRun = np.multiply(velStim,t)
	thetaArray= np.divide(lStim , xRun)
	thetaRun = np.multiply(2 , np.arctan(thetaArray)) # this is the whole angle
	# lSquareRun1 = np.multiply(xStart , np.tan(np.divide(thetaRun1 , 2)))
	lPix1 = np.divide(thetaRun, angPix) # this is the whole length
	thetaDot = -np.divide((lStim/velStim), t**2 + (lStim/velStim)**2)
	thetaDotDeg = np.multiply(thetaDot, 180./math.pi)
	thetaDotPix = np.multiply(thetaDotDeg, 1./angPix)
	thetaDeg = np.multiply(thetaRun, 180./math.pi)
	thetaPix = np.multiply(thetaDeg, 1./angPix)
	lPix = np.multiply(lPix1, scaleRes* 180./math.pi)

	return t, lPix , thetaRun, xRun, dt, lPix1, thetaDotPix, thetaPix

tStimulus, lStimulus, theta, xRun, dt, lStimulus1, thetaDotPix, thetaPix =getParamters(velStim, xStart, fps, lStim, angPix )


yPosPix = []
xPosPix = []
ytimePix = []
xtimePix = []
findTimeDetect = True
continueLoop = True
lStimScale = np.divide(lStimulus, scaleRes)
#print type(lStimScale)
for i in range(np.size(lStimScale)):
	#print i
	if continueLoop == True and lStimScale[i] <= 20 :
		yPosPix.append(lStimScale[i])
		xPosPix.append(tStimulus[i])
		#continueLoop = False 

	# if findTimeDetect == True and tStimulus[i] >= -2:
	# 	ytimePix.append(lStimScale[i])
	# 	xtimePix.append(tStimulus[i])
	# 	findTimeDetect = False

# print yPosPix
# print xPosPix
# print ytimePix
# print xtimePix
#plt.axvline(xPosPix)
#plt.axvline(xtimePix)
#plt.plot(tStimulus, lStimScale)
plt.plot(xPosPix, yPosPix)
plt.show()


lStimulus2 =np.multiply(yPosPix, scaleRes) 
# plt.plot(tStimulus, thetaPix,'--', tStimulus, thetaDotPix)
# plt.show()


def createImage(xFov, yFov , angPix, scaleRes, backgroundColor):
	#outHeight =int(round(yFov / angPix))
	#outWidth =int(round(xFov / angPix)) 	
	outHeight = 20 #Pix
	outWidth = 20 #Pix 

	hImage = int(round(outHeight * scaleRes)) 
	wImage = int(round(outWidth * scaleRes))

	img = np.ones((hImage,wImage,3), np.uint8)*backgroundColor
	#img2 = cv2.imread('background5.png')
	img = cv2.resize(img, (hImage,wImage))
	return img, outHeight, outWidth, hImage, wImage


img, outHeight, outWidth, hImage, wImage = createImage(xFov, yFov, angPix, scaleRes, backgroundColor)


fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
writerOut = cv2.VideoWriter('Test'+str(fps) +'_v_' +str(velStim)+'_l_'+str(lStim)+ '_x_'+ str(xStart)+ '_Sc_'+str(scaleCon)+'.avi', fourcc, fps , (outWidth, outHeight))


print np.size(yPosPix)
print np.size(lStimulus2)

plt.plot(xPosPix, lStimulus2)
# Looming stimulus
for i in np.nditer(lStimulus2) :
	xTopLeft1 = int(round(wImage/2 - (i/2)))
	print xTopLeft1
	yTopLeft1 = int(round(hImage/2 - (i/2)))
	xBottomRight1 = int(round(wImage/2 + (i/2)))
	yBottomRight1 = int(round(hImage/2 + i/ 2))
	figureSquare1 = cv2.rectangle(img,(xTopLeft1,yTopLeft1),(xBottomRight1,yBottomRight1),scaleCon, thickness = cv2.FILLED )
	#cv2.imshow('figureSquare1', img)
	imgResize = cv2.resize(img, (outWidth, outHeight) , interpolation = cv2.INTER_AREA )
	#imgResize = cv2.resize(img, (20, 20) , interpolation = cv2.INTER_AREA )

	cv2.waitKey(int(math.ceil(dt)))
	writerOut.write(imgResize)



writerOut.release()

cv2.destroyAllWindows()


from tempfile import TemporaryFile
outfile = TemporaryFile()
data = []
data = np.column_stack((tStimulus, theta))
np.save(outfile,data)
