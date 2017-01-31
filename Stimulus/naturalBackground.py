import numpy as np
import cv2
import sys, math
import matplotlib.pyplot as plt


# Set stimulus parameters


#velocities = [-0.08, -0.05, -0.1, -0.13, -0.35 , -0.18, -0.25, -0.4]
#velocities = [-0.05, -0.08, -0.1, -0.13, -0.15,-0.18 , -0.2, -0.25, -0.3, -0.35, -0.4]
velocities = [-0.018]
for i in range(len(velocities)):
	velStim = velocities[i]  #m/s velocity of stimulus
	#velStim = -0.08  #m/s velocity of stimulus
	
	lStim = 0.01 #m  half length of stimulus
	xStart = 0.05 #m start approach distance away from specimen 

	# Set locust parameters
	xFov = 180 # degrees, fov locust azimuth
	yFov = 180 # degrees, fov locust elevation
	angPix = 1 # deg/pixel interommatidial angle specimen

	#Set parameters image
	#scaleCon = (100,100,100)
	#scaleCon = (100,100,100)
	scaleCon = (0, 0, 0)
	backgroundColor = 255  # 0 black, 255 white 
	scaleRes = 1  # resolution scale, to downsample later
	fps = 200 # Frame Resolution
	numFrames = 10 # Number of repetition frames at end and beginning 
	#Starting position, 2 is center, 3 left up,, 4 right, down
	xposFrac = 4
	yposFrac = 4 
	#Configure Image 

	def createImage(xFov, yFov , angPix, scaleRes):
		outHeight =int(round(yFov / angPix))
		outWidth =int(round(xFov / angPix)) 	
		hImage = int(round(outHeight * scaleRes)) 
		wImage = int(round(outWidth * scaleRes))

		#img = np.ones((hImage,wImage,3), np.uint8)*backgroundColor
		img2 = cv2.imread('background5.png')
		img = cv2.resize(img2, (hImage,wImage))
		return img, outHeight, outWidth, hImage, wImage


	#Initiate Parameters






	def getParamters(velStim, xStart, fps, lStim, angPix ):
		dt = 1./fps 
		tEnd = 0
		tStart= xStart/velStim
		t = np.arange(tStart, tEnd, dt)
		xRun = np.multiply(velStim,t)
		thetaArray= np.divide(lStim , xRun)
		thetaRun = np.multiply(2 , np.arctan(thetaArray)) # this is the whole angle
		# lSquareRun1 = np.multiply(xStart , np.tan(np.divide(thetaRun1 , 2)))
		lPix1 = np.divide(thetaRun, angPix) # this is the whole length
		lPix = np.multiply(lPix1, scaleRes* 180/math.pi)

		return t, lPix , thetaRun, xRun, dt

	img, outHeight, outWidth, hImage, wImage = createImage(xFov, yFov, angPix, scaleRes)
	tStimulus, lStimulus, theta, xRun, dt  = getParamters(velStim, xStart, fps, lStim, angPix )

	# plt.plot(tStimulus, lStimulus)
	# plt.show()
	# print tStimulus

	# def writerImage(fps, outWidth. outHeight, wImage, hImage, dt, lStimulus, img)

	fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
	writerOut = cv2.VideoWriter('Test'+str(fps) +'_v_' +str(velStim)+'_l_'+str(lStim)+ '_x_'+ str(xStart)+ '_Sc_'+str(scaleCon)+'.avi', fourcc, fps , (outWidth, outHeight))

	# Put in some frames at the beginning:


	# Repetition first frame
	firstFrame = 0
	# for i in range(1,numFrames):
	# 	length = lStimulus[firstFrame] 
	# 	xTopLeft1 = int(round(wImage/2 - (length/2)))
	# 	yTopLeft1 = int(round(hImage/2 - (length /2)))
	# 	xBottomRight1 = int(round(wImage/2 + (length/2)))
	# 	yBottomRight1 = int(round(hImage/2 + length/ 2))
	# 	figureSquare1 = cv2.rectangle(img,(xTopLeft1,yTopLeft1),(xBottomRight1,yBottomRight1),scaleCon, thickness = cv2.FILLED )
	# 	#cv2.imshow('figureSquare1', img)
	# 	imgResize = cv2.resize(img, (outWidth, outHeight) , interpolation = cv2.INTER_AREA )
	# 	cv2.waitKey(int(math.ceil(dt)))
	# 	writerOut.write(imgResize)



	# Looming stimulus
	for i in np.nditer(lStimulus) :
		xTopLeft1 = int(round(wImage/xposFrac - (i/2)))
		yTopLeft1 = int(round(hImage/yposFrac - (i /2)))
		xBottomRight1 = int(round(wImage/xposFrac + (i/2)))
		yBottomRight1 = int(round(hImage/yposFrac + i/ 2))
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



	# yPosPix = []
	# xPosPix = []	
	# continueLoop = True
	# lStimScale = np.divide(lStimulus, scaleRes)
	# #print type(lStimScale)
	# for i in range(np.size(lStimScale)):
	# 	#print i
	# 	if continueLoop == True and lStimScale[i] >= 11 :
	# 		yPosPix.append(lStimScale[i])
	# 		xPosPix.append(tStimulus[i])
	# 		continueLoop = False 

	# print yPosPix
	# print xPosPix
	# plt.axvline(xPosPix)
	# plt.plot(tStimulus, lStimScale)
	# plt.show()



	# Repetition last frame
	# lastFrame = len(lStimulus)-1

	# for i in range(1,numFrames):
	# 	length = lStimulus[lastFrame] 
	# 	xTopLeft1 = int(round(wImage/2 - (length/2)))
	# 	yTopLeft1 = int(round(hImage/2 - (length /2)))
	# 	xBottomRight1 = int(round(wImage/2 + (length/2)))
	# 	yBottomRight1 = int(round(hImage/2 + length/ 2))
	# 	figureSquare1 = cv2.rectangle(img,(xTopLeft1,yTopLeft1),(xBottomRight1,yBottomRight1),scaleCon, thickness = cv2.FILLED )
	# 	#cv2.imshow('figureSquare1', img)
	# 	imgResize = cv2.resize(img, (20, 20) , interpolation = cv2.INTER_AREA )
		
	# 	#imgResize = cv2.resize(img, (outWidth, outHeight) , interpolation = cv2.INTER_AREA )
	# 	cv2.waitKey(int(math.ceil(dt)))
	# 	writerOut.write(imgResize)


