import numpy as np 
import matplotlib.pyplot as plt
import sys
import scipy 
from scipy.stats import linregress 

# Set stimulus parameters experiment
velStim = -0.25  #m/s velocity of stimulus
lStim = 0.01 #m  half length of stimulus
xStart = 0.05 #m start approach distance away from specimen 
fps = 200 # Frame Resolution
angPix = 4.5 # deg/pixel interommatidial angle specimen

#Name layers:
#nameFile = 'HueData' 
#nameFile = 'dataV025.dat' 
groupA1 = 'Hue'
groupB1 = ' Edge Detection '




#Imprt data from iqr neural network

def importData(nameFile):
	data= np.genfromtxt(nameFile, delimiter = ';')
	data= data[1:,:] #exclude first row nomenclature
	return data

# Read text file sampling data, extract ID
def findID(nameFile):
	IDNums=[]
	IDNames=[]
	IDMat = []

	with open(nameFile) as f:
		for line in f:
			line = line.strip()
			if '#' in line:
				printName = True
				IDFind=line.split('"')
				IDNumber = IDFind[2]
				IDNumber = IDNumber[1:]
				IDName=IDFind[1]

				if IDName in IDNames: 
				 	printName = False
				if printName == True: 
					IDNums.append(IDNumber)
					IDNames.append(IDName)
					lengthNum=int(len(IDNumber))
					IDMat.append ([IDNumber, IDName,lengthNum])
					IDName2 = IDName
		IDMat= np.asarray(IDMat)		
		IDMat =IDMat[np.sort(IDMat[:,2].argsort())[::-1]]
		IDNames = IDMat[:,1]
		IDNums = IDMat[:,0]
	return IDNums, IDNames


# Rewrite IDNumber with IDName
def rewriteCycleLine(nameFile):	
	IDNums, IDNames = findID(nameFile)
	with open(nameFile) as f:
		for line in f:
			line = line.strip()
			if 'cycle' in line:
				IDLine = line.split(';')
				for i in range(len(IDNames)):
					IDLine= [w.replace(str(IDNums[i]), IDNames[i]) for w in IDLine]
					IDLine= [w.replace('RangeAverage', 'RaAv') for w in IDLine]
	return IDLine
			
# Create custom defined plots
def createPlot(groupNameA, inhibitoryA, excitatoryA, cellPotentialA, activityA, averageA, neuronsA,nameFile):
	IDLine = rewriteCycleLine(nameFile)
	data = importData(nameFile)
	groupPlot = []
	partA = []
	partA2 = []
	time= data[:,0]
	if inhibitoryA == 1:
		partA.append('_inhIn_')
	if excitatoryA == 1 :
		partA.append('_excIn_')
	if cellPotentialA == 1:
		partA.append('_vm_')
	if activityA== 1:
		partA.append('_act_')
	if averageA == 1:
		partA2 = 'RaAv'
	showPlots = False

	for i in range(len(partA)):
		nameTrue=groupNameA + partA[i] + partA2
		if nameTrue in IDLine:
			columnGroup = data[:, IDLine.index(groupNameA + partA[i] + partA2)]
			groupPlot = plt.plot(time,columnGroup, label=groupNameA + partA[i] + partA2)

			#plt.legend()
			showPlots = True
		else:
			print(nameTrue + ' is not in data list')
	if showPlots == True :
		plt.figure()
		plt.show(block = False)	
	return groupPlot, showPlots

#Make Plots, all data
def plotAllData(nameFile) :
	IDLine = rewriteCycleLine(nameFile)
	data = importData(nameFile)
	time= data[:,0] 
	for i in range(1, len(IDLine),1):
		allPlots= plt.plot(time, data[:,i], label =(str(IDLine[i])))
		plt.legend()
	return allPlots



# partA = ['_act_', '_inhIn_']
# partA2 = 'RaAv'
# allGroups = False 
# groupNames= [groupA1,groupB1]


def definePlots(nameFile, partA, partA2, allGroups, groupNames, scale, plotFile1, xTimeCol, namePlot):
	data = importData(nameFile)
	# print nameFile
	lOverV, v, l, typeStim = extractValuesFileName(nameFile)
	IDLine = rewriteCycleLine(nameFile)
	#scale =  (float(0.05)/float(v))/float(xTimeCol[0])
	time = data[:,0]* scale
	groupPlot = []

	IDLine2 = []
	if allGroups == True:
		groupNames= ['']
	if plotFile1 == True :

		
		for j in range(len(groupNames)):
			groupName = groupNames[j]

			for i in range(1, len(IDLine),1):
				
				for k in range(len(partA)):
					nameTrue=groupName + partA[k] + partA2
					if k == 0:
						colorPlot = 'b'
					if k ==1: 
						colorPlot = 'r'
					if k == 2:
						colorPlot = 'g'	
					if k>2:
						colorPlot = 'b'			
					if nameTrue in IDLine[i]:
						ID = IDLine[i]
						IDLine2.append(ID)
						columnGroup = data[:, i]
						if plotFile1 == True:
							groupPlot = plt.plot(time,columnGroup,colorPlot, label=(str(groupName)+str(partA[k])+'_l/v_'+str(lOverV)))
		plt.xlabel('Time [ms]', fontsize= 20)
		plt.ylabel('Group Activity [V]', fontsize= 20)
		plt.title('LGMD Activity - '+namePlot, fontsize= 24)
		plt.legend()
		plt.show()
	if plotFile1 ==False:
		groupPlot = False

	print 'definePlot'				
	return groupPlot

def extractValuesFileName(nameFile):
	nameFileSplit = []
	nameFiles = np.array(nameFile)
	if np.size(nameFile)== 1:
		nameFiles = [nameFile]
	for i in range(np.size(nameFiles)):
		#print i
		nameFileMat = nameFiles[i].split('_')
		#print nameFileMat
		if 'v' in nameFileMat: 
			v=nameFileMat[nameFileMat.index('v')+1]
		if 'l' in nameFileMat:
			l=nameFileMat[nameFileMat.index('l')+1]
		if 'pixDet' in nameFileMat:
			pixDet = nameFileMat[nameFileMat.index('pixDet')+1]
		if 'type' in nameFileMat:
			typeStim=nameFileMat[nameFileMat.index('type')+1]

		if 'v' and 'l' in nameFileMat:
			lOverV= np.divide(float(l),float(v))
		 	lOverV	="{0:.2f}".format(lOverV)
		nameFileSplit.append(nameFileMat)



	return  lOverV, v, l, typeStim

def extractFiles(filePath):
	import os
	nameFiles =[]
	for file in os.listdir(filePath):
	    if file.endswith(".dat"):
	        #print(file)
	       	nameFiles.append(file)
	return nameFiles



def plotTimeCollision(nameFile, groupA1, partA, partA2, threshold, plotTimeCol1):

	#lOverV, v, l, typeStim = extractValuesFileName(nameFile)
	#print v
	data = importData(nameFile)
	IDLine = rewriteCycleLine(nameFile)
	time = data[:,0]
	data2 = data[:, IDLine.index(groupA1 + partA+ partA2)]
	#threshold = 0.2
	#print type(threshold)

	if threshold >0.001:
		xPos = (np.r_[True, data2[1:] < data2[:-1]] & np.r_[data2[:-1] < data2[1:], True])* (data2<threshold) * time 
	
	if threshold == 0.001:
		xPos = (np.r_[True, data2[1:] < data2[:-1]])* (data2<threshold) * time 

	xTimeCol= []



	for i in np.nditer(xPos):
		if i> 0 :
			xTimeCol.append(int(i))
			#scaleI = scaleTimeReality(nameFile, xTimeCol)
			#j = i*scaleI
			if plotTimeCol1 == True:
				timeCol = plt.axvline(i, color ='k', linewidth=2.5, linestyle = '--', label='time of collision')
				#plt.legend('collision time')
			if plotTimeCol1 == False:
				timeCol = False

	return timeCol, xTimeCol, 




def setThresholds(nameFiles, groupA1, partA, partA2):
	thresholds = []
	#print nameFiles
	#nameFiles = extractFiles(filePath)
	for i in range(len(nameFiles)):
		nameFile=nameFiles[i]
		data = importData(nameFile)
		IDLine = rewriteCycleLine(nameFile)
		time = data[:,0]
		data2 = data[:, IDLine.index(groupA1 + partA + partA2)]
		plt.plot(time, data2, label = nameFile)
		plt.legend()
		plt.show()
		threshold = float(raw_input('threshold: '))
		thresholds.append(threshold)
		plotThres= True
	return thresholds


def setThresholdsSpike(nameFiles):
	thresSpike = []
	#print nameFiles
	#nameFiles = extractFiles(filePath)
	for i in range(len(nameFiles)):
		nameFile=nameFiles[i]
		data = importData(nameFile)
		IDLine = rewriteCycleLine(nameFile)
		time = data[:,0]
		data2 = data[:, IDLine.index( 'LGMD WTA' + '_act_' + 'RaAv')]
		plt.plot(time, data2, label = nameFile)
		plt.legend()
		plt.show()
		threshold = float(raw_input('threshold spike: '))
		thresSpike.append(threshold)
		plotThres= True
	return thresSpike


def findLGMDSpike(nameFile, plotLGMDspike, thresholdSpike, scale1):
	data = importData(nameFile)
	IDLine = rewriteCycleLine(nameFile)
	time = data[:,0]#*scale1
	data3 = data[:, IDLine.index('LGMD WTA' + '_act_' + 'RaAv')]
	threshold = float(thresholdSpike)
	#xPos = (np.r_[True, data3[1:] > data3[:-1]] & np.r_[data3[:-1] > data3[1:], True])* (data3<threshold) * time 
	xPos = (np.r_[True, data3[1:] > data3[:-1]])* (data3>threshold) * time 
	xTimeSpike = []
	if np.sum(xPos)>0:
		for i in np.nditer(xPos):
			if i> 0 :
				xTimeSpike.append(int(i))
				if plotLGMDspike == True: 
					timeSpike = plt.axvline(i, color ='b', linewidth=2.5, linestyle = '--')
				if plotLGMDspike == False :
					timeSpike = False

	elif np.sum(xPos)==0:
		xTimeSpike = False
		timeSpike = False
	return timeSpike, xTimeSpike



	
def createPlotFiles(filePath, partA, partA2, allGroups, groupNames, scale, plotFile1, plotTimeCol1, plotLGMDspike, plotScatter, setThresholdsMan, thresholdHue,  thresholdSpike,setThresholdsSpikeMan, typeTrue, setColor, namePlot):
	nameFiles1 = extractFiles(filePath)
	nameFiles = []


	for i in range(len(nameFiles1)):
		if typeTrue in nameFiles1[i]:
			nameFile2= nameFiles1[i]
			nameFiles.append(nameFile2)
	print nameFiles
	if setThresholdsMan == True:
		thresholds = setThresholds(nameFiles, 'Hue', '_act_','RaAv')
	if setThresholdsMan == False :
		thresholds= thresholdHue
		if np.size(thresholdHue) == 1 :
			number = len(nameFiles)
			thresholds = np.ones(number)*thresholdHue

	if setThresholdsSpikeMan == True:
		thresSpike = setThresholdsSpike(nameFiles)
	
	if setThresholdsSpikeMan == False : 
		thresSpike = thresholdSpike
		if np.size(thresholdSpike) ==1 :
			thresSpike = np.ones(len(nameFiles))*thresholdSpike

	xTime2 = []
	spikeTime = []
	xlv = []
	ysc = []
	zst =[]
	frameColission = []
	tryout = []	
	if plotFile1 ==True:
		fig = plt.figure()
		plot = fig.add_subplot(111)
		plot.tick_params(labelsize = 18)

	for i in range(len(nameFiles)):
		nameFile=nameFiles[i]
		if typeTrue in nameFile:
			print nameFile
			threshold = thresholds[i]
			thresholdSpikes = thresSpike[i]		
			plotTimeCol, xTimeCol = plotTimeCollision(nameFile, 'Hue', '_act_','RaAv', threshold, plotTimeCol1)
			#scale = scaleTimeReality(nameFile, xTimeCol)
			scale = 1		
			plotFile = definePlots(nameFile, partA, partA2, allGroups, groupNames, scale, plotFile1, xTimeCol, namePlot)
			timeSpike, xTimeSpike = findLGMDSpike(nameFile, plotLGMDspike, thresholdSpikes, scale)
			spikeTime.append(xTimeSpike)
			frameColission.append(xTimeCol)
			lOverV, v, l, typeStim = extractValuesFileName(nameFile)
			if xTimeSpike !=False :
				xloverV, ySpike2col, zScaled = spike2colision(xTimeSpike, xTimeCol, lOverV, v, scale) #, plotScatter)
				xlv.extend(xloverV)
				ysc.extend(ySpike2col)
				zst.extend(zScaled)
			xTime2.extend(xTimeCol)
			
		
		#plt.scatter(xlv,ysc)
	 
	#plotSpikeCol, plotRegression = linearRegress(xlv,ysc, plotScatter, typeTrue, setColor, '')
	
	plotSpikeCol, plotRegression = linearRegress(xlv,zst, plotScatter, typeTrue, setColor, '', namePlot)

	# Plot Scatter		
	

	
	return plotFile, plotTimeCol, xTime2, plotSpikeCol, plotRegression, xlv, ysc, xTimeSpike, frameColission

def linearRegress(xlv, ysc, plotScatter, typeTrue, setColor, difName, namePlot):

	xlv= np.array(xlv)
	ysc= np.array(ysc) 
	slope, intercept,rValue,a4,a5 =linregress(xlv,ysc)
	fity = slope * xlv + intercept
	rValue2	="{0:.3f}".format(rValue)
	slope2	="{0:.3f}".format(slope)
	intercept2	="{0:.3f}".format(intercept)
	if plotScatter == True:
		plotSpikeCol = plt.scatter(xlv,ysc, label =namePlot, color = setColor)
		plotRegression = plt.plot(xlv, fity, color = setColor, label =( ' y='+slope2+'*x+'+intercept2+'  r='+ rValue2 + difName))
		plt.xlabel('l/v [ms]' , fontsize= 22)
		plt.ylabel('time to collision [ms]' , fontsize= 22)
		plt.axhline(0, color ='k', linewidth=2.5, linestyle = '--' )
		#plt.legend()
		plt.title('LGMD TTC response', fontsize= 30)
		plt.legend()

		#plt.legend(handles=[plotSpikeCol, plotRegression], loc=4)
	if plotScatter == False:
		plotSpikeCol = False
		plotRegression = False 
	print 'scatter On'

	return plotSpikeCol, plotRegression


def spike2colision(timeSpike, xTimeCol, lOverV, v, scale1):#, plotScatter):
	x = []
	y = []
	z= []

	for j in range(len(timeSpike)):
		if j<1:
			x.append(float(lOverV)*1000)
			spike2col = (xTimeCol[0] - timeSpike[j])
			 
			scaledTime = (float(spike2col) * 0.05/float(v)) /float(xTimeCol[0])*1000
			#scaledTime = float(spike2col)* scale1*1000
			y.append(float(spike2col)) 
			z.append(float(scaledTime))
	return 	x,y, z #, spike2colPlot	

def scaleTimeReality(nameFile, xTimeCol):
	xDistance = 0.05
	lOverV, v, l, typeStim = extractValuesFileName(nameFile)
	scale =  (float(xDistance)/float(v))/float(xTimeCol[0])

	return scale 









	# fit = np.polyfit(xlv,ysc,1)
	# fit_fn = np.poly1d(fit)
	# plt.plot(xlv,ysc,'yo', xlv, fit_fn(xlv))





	# if len(typeTrue) > 1:
	# 	for i in range(len(nameFiles)):
	# 		print i
	# 		if typeTrue in nameFiles[i]:
	# 			nameFiles2 = []
	# 			nameFiles2.append(nameFiles[i])
	# 			print nameFiles2

	# 	nameFiles = nameFiles2	

#v, l, pixDet, typeStim, lOverV = extractValuesFileName(nameFile)

# allGroups = True
# plt.show()
# if allGroups == True:
# 	groupName = ''
# 	for i in range(1, len(IDLine),1):
# 		nameTrue=groupName + partA + partA2
		
# 		if nameTrue in IDLine[i]:
# 			ID = IDLine[i]
# 			IDLine2.append(ID)
# 			columnGroup = data[:, i]
# 			groupPlot = plt.plot(time,columnGroup, label=str(IDLine[i]))
# 			plt.legend()



	# for i in range(len(partA)):
	# 	if nameTrue in IDLine:
	# 		columnGroup = data[:, IDLine.index(partA[i] + partA2)]
	# 		groupPlot = plt.plot(time,columnGroup, label=str(IDLine[columnGroup]))
	# 		plt.legend()
	# 		showPlots = True
	# 	else:
	# 		print (nameTrue + ' is not in data list')
	# if showPlots == True :
	# 	plt.figure()
	# 	plt.show(block = False)	
	# return groupPlot, showPlots	


#create plots to compare
#name. inhibitory, excitatory, vm, act, average, neuron


# # IDNames = IDMat[:,1]
# # IDNums = IDMat[:,0]
# print IDNums



# IDMat= np.asarray(IDMat)


#Mat3 =IDMat2[np.argsort((IDMat2[:,2]))]
# IDMat =IDMat[IDMat[:,2].argsort()[::-1]]
# print IDMat
# # IDMat = []
# IDMat.append([])
# IDMa
# IDMat[0].append(IDNums)
# IDMat[1].append(IDNames)
# print IDMat 
# allPlots = plotAllData(IDLine, data)
# plt.show()
# groupPlotA= createPlot(' Edge Detection ', 1, 0,0, 0 ,1 ,0, data)
# groupPlotB= createPlot('Hue' , 0, 0,1, 0 ,1 ,0, data)
# plt.show()



# if foundA1 == True :
			# groupA1Exc_avg=data[1:-1, IDLine.index(groupA1 + '_excIn_RangeAverage')]
			# groupA1Vm_avg= data[1:-1, IDLine.index(groupA1 + '_vm_RangeAverage')]



# Calculate values stimulus
# from sys import path	
# path.append('/home/atilla/Documents/Test/Neural_Network/Stimulus/')
# from ommatidiaStimulus2 import getParamters
# tStim, lPixStim , thetaStim, xStim, dt = getParamters(velStim, xStart, fps, lStim, angPix)
# thetaNorm = (thetaStim - np.amin(thetaStim)) / (np.amax(thetaStim) - np.amin(thetaStim)) 
# #Scale time back to experiment
# #Not certain whether this scale is correct, i
# tStimScale = (tStim*len(data))+len(data)

# Define rows and columns of neural network data
#have to include to read file, so it does it automatically, search for the string Inh, etc and then take that column
# is in the firts row of the text file, so should be easy to implement
#also see what happens to data if more groups are added to the datasampling, how to classify



# g =open(nameFile, 'r')
# filedata = g.read()
# g.close()

# #for i in len(IDNums):
# newdata = filedata.replace(IDNums[1], IDNames[1])
# g=open('TextOut', 'w')
# g.write(newdata)
# g.close()

# if foundA1 == True:
# 	newdata = filedata.replace(IDGroupA1, groupA1)
# if foundB1 == True:
# 	g = open(nameFile, 'w')
# 	g.write(newdata)
# 	g.close()
#will probably produce an error if not there, 			
