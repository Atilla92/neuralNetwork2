
import numpy as np
import cv2
import sys, math
import matplotlib.pyplot as plt
from sys import path
from CalculateResolution import getParamters, plotPixels
path.append('/home/atilla/Documents/Test/Neural_Network2/Stimulus/Data/')
import dataAnalysis as datAn
filePath= '/home/atilla/Documents/Test/Neural_Network2/Stimulus/Data/'


#Define Plots
groupA1 = 'Hue'
groupB1 = ' Edge Detection '
groupC1 = 'ON cell' 
groupC2 = ' OFF cell'
groupD1 = 'EMD Up'
groupD2 = 'EMD Down'
groupD3 = 'EMD Left'
groupD4 = 'EMD Right' 
groupE1 = 'WTA Excit'
groupE2 = 'WTA Inhib'
groupF1 = 'LGMD WTA'
groupNames= [groupA1, groupF1]#, groupE1]# groupC1, groupC2]
partA =['_act_']# ['_inhIn_']
partA2 = 'RaAv'
allGroups = False 
scale = 1

######Plot 1#######
setColor = 'b'
plotTimeCol = False 
plotFile = False
plotLGMDspike = True
plotScatter =False
setThresholdsMan =False
setThresholdsSpikeMan = False
thresholdHue = 0.1
thresholdSpike = 0.0015
typeTrue = 'NatBG'
namePlot = 'White bg. Black Square, th=0.2 40'
velStim = -22 
xStart =70 
lStim = 0.45
fps = 30.
outHeight = 40
scaleRes = 10
pixelRes = 7
angPix =1./pixelRes
pixelDetect = 5
plotDefineAgain, plotTimeCol, xTime2, plotSpikeCol, plotRegression2, xlv2, ysc2, xFrameSpike, frameCollision= datAn.createPlotFiles(filePath, partA, partA2, allGroups, groupNames, scale, plotFile, plotTimeCol, plotLGMDspike, plotScatter, setThresholdsMan, thresholdHue,  thresholdSpike, setThresholdsSpikeMan, typeTrue, setColor, namePlot)
yPosPix,xPosPix, theta,  dt , tStimulus, tPix5=getParamters(velStim, xStart, fps, lStim, angPix,scaleRes, outHeight, pixelDetect )
plt.show()
line1, line2 , line3= plotPixels(xFrameSpike, fps, xPosPix , yPosPix, tPix5)
plt.show()

######Plot 12######
setColor = 'b'
plotTimeCol = False 
plotFile = False
plotLGMDspike = True
plotScatter =False
setThresholdsMan =True
setThresholdsSpikeMan = True
thresholdHue = 0.1
thresholdSpike = 0.0015
typeTrue = 'NatBackG'
namePlot = 'White bg. Black Square, th=0.2 40'
velStim = -22 
xStart =70 
lStim = 0.45
fps = 60.
outHeight = 40
scaleRes = 10
pixelRes = 7
angPix =1./pixelRes
pixelDetect = 5
plotDefineAgain, plotTimeCol, xTime2, plotSpikeCol, plotRegression2, xlv2, ysc2, xFrameSpike, frameCollision= datAn.createPlotFiles(filePath, partA, partA2, allGroups, groupNames, scale, plotFile, plotTimeCol, plotLGMDspike, plotScatter, setThresholdsMan, thresholdHue,  thresholdSpike, setThresholdsSpikeMan, typeTrue, setColor, namePlot)
yPosPix,xPosPix, theta,  dt , tStimulus, tPix5=getParamters(velStim, xStart, fps, lStim, angPix,scaleRes, outHeight, pixelDetect )
plt.show()
line1, line2 , line3= plotPixels(xFrameSpike, fps, xPosPix , yPosPix, tPix5)
plt.show()

