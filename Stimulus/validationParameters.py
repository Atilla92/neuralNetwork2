

import numpy as np
import cv2
import sys, math
import matplotlib.pyplot as plt
from sys import path


# Set requirements worst case scenario
timeReq = 2 #s
vMax = 11 #m/s
xFov = 180 # degrees
yFov = 180 # degrees
pixRes = 5 # units
xSize = 0.450 # m
ySize = 0.150 # m 


#Estimate parameters
vApproach = vMax *2 # m/s
xApproach = vApproach * timeReq # m 
xAngularRes = np.arctan(xSize/xApproach) * 180/math.pi *2 # degrees
yAngularRes = np.arctan(ySize/xApproach)* 180/math.pi *2# degrees
xResReq = xFov/xAngularRes * pixRes
yResReq = yFov/yAngularRes * pixRes
print xAngularRes
print yAngularRes
print xResReq
print yResReq