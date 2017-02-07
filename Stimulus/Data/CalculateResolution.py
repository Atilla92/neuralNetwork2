import numpy as np
import cv2
import sys, math
import matplotlib.pyplot as plt
from sys import path

#path.append('/home/atilla/Documents/Test/Neural_Network/Stimulus/')

#import naturalBackground.py
pixelDetect = 7
velStim = -22 #m/s velocity of stimulus
	#velStim = -0.08  #m/s velocity of stimulus
	
lStim = 0.45 #m  half length of stimulus
xStart = 70 #m start approach distance away from specimen 

# Set locust parameters
xFov = 180 # degrees, fov locust azimuth
yFov = 180 # degrees, fov locust elevation
angPix = 1./pixelDetect # deg/pixel interommatidial angle specimen

scaleCon = (0, 0, 0)
backgroundColor = 255  # 0 black, 255 white 
scaleRes = 10  # resolution scale, to downsample later
fps = 60. # Frame Resolution
numFrames = 10 # Number of repetition frames at end and beginning 
#Starting posit


outHeight = 40 #Pix
outWidth = 40 #Pix 

#create a video, set True
createVideo =True
def createImage(xFov, yFov , angPix, scaleRes, backgroundColor, outWidth, outHeight):
	#outHeight =int(round(yFov / angPix))
	#outWidth =int(round(xFov / angPix)) 	
	hImage = int(round(outHeight * scaleRes)) 
	wImage = int(round(outWidth * scaleRes))

	#img = np.ones((hImage,wImage,3), np.uint8)*backgroundColor
	img2 = cv2.imread('background5.png')
	img = cv2.resize(img2, (hImage,wImage))
	return img, outHeight, outWidth, hImage, wImage


def getParamters(velStim, xStart, fps, lStim, angPix, scaleRes, outHeight, pixelDetect ):
	yPosPix = []
	xPosPix = []
	ytimePix = []
	xtimePix = []
	tPix5 = []
	continueLoop = True
	find5 = True
	dt = 1./fps 
	tEnd = 0
	tStart= float(xStart/velStim)
	t = np.arange(tStart, tEnd, dt)
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
	lStimScale = np.divide(lPix, scaleRes)
	for i in range(np.size(lStimScale)):
		#print i
		if continueLoop == True and lStimScale[i] <= outHeight :
			yPosPix.append(lStimScale[i])
			xPosPix.append(t[i])
		if find5 == True and lStimScale[i]>= pixelDetect:
			tPix5.append(t[i])
			find5 = False


	#return t, lPix , thetaRun, xRun, dt, lPix1, thetaDotPix, thetaPix
	return yPosPix, xPosPix, thetaRun, dt, t, tPix5



yPosPix,xPosPix, theta,  dt , tStimulus, tPix5=getParamters(velStim, xStart, fps, lStim, angPix,scaleRes, outHeight, pixelDetect )

#plt.plot(xPosPix, yPosPix)
#plt.show()

  
if createVideo == True:
	img, outHeight, outWidth, hImage, wImage = createImage(xFov, yFov, angPix, scaleRes, backgroundColor, outWidth, outHeight)
	fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
	writerOut = cv2.VideoWriter('WCS_NaturalBG_Grey'+'_fps_'+str(fps) +'_v_' +str(velStim)+'_l_'+str(lStim)+ '_x_'+ str(xStart)+ '_Sc_'+str(scaleCon)+'_Res_'+str(outHeight) +'_PixRes_'+str(pixelDetect)+'.avi', fourcc, fps , (outWidth, outHeight))

	lStimulus2 =np.multiply(yPosPix, scaleRes)
	for i in np.nditer(lStimulus2) :
		xTopLeft1 = int(round(wImage/2 - (i/2)))
		yTopLeft1 = int(round(hImage/2 - (i/2)))
		xBottomRight1 = int(round(wImage/2 + (i/2)))
		yBottomRight1 = int(round(hImage/2 + i/ 2))
		figureSquare1 = cv2.rectangle(img,(xTopLeft1,yTopLeft1),(xBottomRight1,yBottomRight1),scaleCon, thickness = cv2.FILLED )
		#cv2.imshow('figureSquare1', img)
		
		imgResize1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		imgResize = cv2.resize(imgResize1, (outWidth, outHeight) , interpolation = cv2.INTER_AREA )
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


print len(yPosPix)

path.append('/home/atilla/Documents/Test/Neural_Network2/Stimulus/Data/')
import dataAnalysis as datAn


filePath= '/home/atilla/Documents/Test/Neural_Network2/Stimulus/Data/'





def plotPixels(xFrameSpike, fps, xPosPix , yPosPix, tPix5):
	xTimeSpike = np.divide(xFrameSpike, fps)
	for i in range(np.size(xFrameSpike)):
		j = xFrameSpike[i]-1
		if j< np.size(xPosPix):
			timeSpike = xPosPix[j]
			pixSize = yPosPix[i]
			pixplotSpike = plt.axvline(timeSpike)
			pixPlotStim = plt.plot(xPosPix, yPosPix)
			if i == 0:
				timeBetween= timeSpike-tPix5
				print timeBetween
				plot5 = plt.axvline(tPix5, color = 'r')
	return pixPlotStim, pixplotSpike, plot5

