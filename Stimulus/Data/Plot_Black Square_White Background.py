
import numpy as np
import cv2
import sys, math
import matplotlib.pyplot as plt
from sys import path
from CalculateResolution import getParamters, plotPixels
path.append('/home/atilla/Documents/Test/Neural_Network2/Stimulus/Data/')
import dataAnalysis as datAn



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

plotTimeCol = True 
plotFile = False
plotLGMDspike = True
plotScatter =False
setThresholdsMan =False
setThresholdsSpikeMan = False
setColor = 'b'
thresholdHue = [0.01]
thresholdSpike = 0.004
typeTrue = 'Square'
namePlot = 'White bg. Black Square, th=0.2 '
xStart =50 
fps = 200.
angPix = 0.2
pixelDetect = 5

plotDefineAgain, plotTimeCol, xTime2, plotSpikeCol, plotRegression2, xlv2, ysc2, xFrameSpike, frameCollision= datAn.createPlotFiles(filePath, partA, partA2, allGroups, groupNames, scale, plotFile, plotTimeCol, plotLGMDspike, plotScatter, setThresholdsMan, thresholdHue,  thresholdSpike, setThresholdsSpikeMan, typeTrue, setColor, namePlot)
yPosPix,xPosPix, theta,  dt , tStimulus, tPix5=getParamters(velStim, xStart, fps, lStim, angPix,scaleRes, outHeight, pixelDetect )
plt.show()
line1, line2, line3 = plotPixels(xFrameSpike, fps, xPosPix , yPosPix, tPix5)
plt.show()

##### Plot 2###### 

plotTimeCol = False 
plotFile = False
plotLGMDspike = True
plotScatter =False
setThresholdsMan =False
setThresholdsSpikeMan = False
setColor = 'b'
thresholdHue = 0.01
thresholdSpike = 0.0004
typeTrue = 'squareBig'
namePlot = 'White bg. Black Square, th=0.2 40'
xStart =50 
fps = 200.
angPix = 0.2
pixelDetect = 5
plotDefineAgain, plotTimeCol, xTime2, plotSpikeCol, plotRegression2, xlv2, ysc2, xFrameSpike, frameCollision= datAn.createPlotFiles(filePath, partA, partA2, allGroups, groupNames, scale, plotFile, plotTimeCol, plotLGMDspike, plotScatter, setThresholdsMan, thresholdHue,  thresholdSpike, setThresholdsSpikeMan, typeTrue, setColor, namePlot)
yPosPix,xPosPix, theta,  dt , tStimulus, tPix5=getParamters(velStim, xStart, fps, lStim, angPix,scaleRes, outHeight, pixelDetect )
plt.show()
line1, line2, line3 = plotPixels(xFrameSpike, fps, xPosPix , yPosPix, tPix5)
plt.show()


##### Plot 3 ######
plotTimeCol = False 
plotFile = False
plotLGMDspike = True
plotScatter =False
setThresholdsMan =False
setThresholdsSpikeMan = False
setColor = 'b'
thresholdHue = 0.01
thresholdSpike = 0.0004
typeTrue = 'SquBig'
namePlot = 'White bg. Black Square, th=0.2 40'
xStart =50 
fps = 200.
pixelRes = 7
angPix =1./pixelRes
pixelDetect = 5
plotDefineAgain, plotTimeCol, xTime2, plotSpikeCol, plotRegression2, xlv2, ysc2, xFrameSpike, frameCollision= datAn.createPlotFiles(filePath, partA, partA2, allGroups, groupNames, scale, plotFile, plotTimeCol, plotLGMDspike, plotScatter, setThresholdsMan, thresholdHue,  thresholdSpike, setThresholdsSpikeMan, typeTrue, setColor, namePlot)
yPosPix,xPosPix, theta,  dt , tStimulus, tPix5=getParamters(velStim, xStart, fps, lStim, angPix,scaleRes, outHeight, pixelDetect )
plt.show()
line1, line2 , line3= plotPixels(xFrameSpike, fps, xPosPix , yPosPix, tPix5)
plt.show()



##### Plot 4 ####### 
plotTimeCol = False 
plotFile = False
plotLGMDspike = True
plotScatter =False
setThresholdsMan =False
setThresholdsSpikeMan = False
setColor = 'b'
thresholdHue = 0.001
thresholdSpike = 0.001
typeTrue = 'squaBig'
namePlot = 'White bg. Black Square, th=0.2 40'
xStart =70 
fps = 200.
pixelRes = 7
angPix =1./pixelRes
pixelDetect = 5

plotDefineAgain, plotTimeCol, xTime2, plotSpikeCol, plotRegression2, xlv2, ysc2, xFrameSpike, frameCollision= datAn.createPlotFiles(filePath, partA, partA2, allGroups, groupNames, scale, plotFile, plotTimeCol, plotLGMDspike, plotScatter, setThresholdsMan, thresholdHue,  thresholdSpike, setThresholdsSpikeMan, typeTrue, setColor, namePlot)
plt.show()
yPosPix,xPosPix, theta,  dt , tStimulus, tPix5=getParamters(velStim, xStart, fps, lStim, angPix,scaleRes, outHeight, pixelDetect )
line1, line2, line3 = plotPixels(xFrameSpike, fps, xPosPix , yPosPix, tPix5)
plt.title('200 Hz Stimulus')
plt.show()


##### Plot 5 #######
plotTimeCol = False 
plotFile = False
plotLGMDspike = True
plotScatter =False
setThresholdsMan =False
setThresholdsSpikeMan = False
setColor = 'b'
thresholdHue = 0.05
thresholdSpike = 0.001
typeTrue = 'lowFps'
namePlot = 'White bg. Black Square, th=0.2 40'
xStart =70 
fps = 60.
pixelRes = 7
angPix =1./pixelRes
pixelDetect = 5
plotDefineAgain, plotTimeCol, xTime2, plotSpikeCol, plotRegression2, xlv2, ysc2, xFrameSpike, frameCollision= datAn.createPlotFiles(filePath, partA, partA2, allGroups, groupNames, scale, plotFile, plotTimeCol, plotLGMDspike, plotScatter, setThresholdsMan, thresholdHue,  thresholdSpike, setThresholdsSpikeMan, typeTrue, setColor, namePlot)
plt.show()
yPosPix,xPosPix, theta,  dt , tStimulus, tPix5=getParamters(velStim, xStart, fps, lStim, angPix,scaleRes, outHeight, pixelDetect )
line1, line2, line3 = plotPixels(xFrameSpike, fps, xPosPix , yPosPix, tPix5)
plt.title('60 Hz Stimulus')
plt.legend()
plt.show()


#### Plot 6 #### 

plotTimeCol = False 
plotFile = False
plotLGMDspike = True
plotScatter =False
setThresholdsMan =False
setThresholdsSpikeMan = False
setColor = 'b'
thresholdHue = 0.15
thresholdSpike = 0.001
typeTrue = 'squ30'
namePlot = 'White bg. Black Square, th=0.2 40'
xStart =70 
fps = 30.
pixelRes = 7
angPix =1./pixelRes
pixelDetect = 5
plotDefineAgain, plotTimeCol, xTime2, plotSpikeCol, plotRegression2, xlv2, ysc2, xFrameSpike, frameCollision= datAn.createPlotFiles(filePath, partA, partA2, allGroups, groupNames, scale, plotFile, plotTimeCol, plotLGMDspike, plotScatter, setThresholdsMan, thresholdHue,  thresholdSpike, setThresholdsSpikeMan, typeTrue, setColor, namePlot)
plt.show()
yPosPix,xPosPix, theta,  dt , tStimulus, tPix5=getParamters(velStim, xStart, fps, lStim, angPix,scaleRes, outHeight, pixelDetect )
line1, line2, line3 = plotPixels(xFrameSpike, fps, xPosPix , yPosPix, tPix5)
plt.title('30 Hz Stimulus')
plt.legend()
plt.show()
