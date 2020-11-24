# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:46:52 2020

@author: Hsin-Yi
"""
import numpy as np


sorting = np.sort(snr_2)
sorting = sorting[::-1]

extreme = np.zeros((1031, 2))
j=0
while j <1031:
    xx = np.where(snr_2 == sorting[j])
    for k in range(len(xx[0])):
        extreme[j, 0] = res[0][xx[0][k]]
        extreme[j, 1] = res[1][xx[0][k]]
        j=j+1
extreme = extreme.astype(int)

        
image = np.zeros((1024, 1024))
image[:] =0

for j in range(1031):
    image[extreme[j,0], extreme[j,1]] = snr[extreme[j,0], extreme[j,1]]
image[image < 10**21] = 0
image[image > 10**22] = 10
image[image > 10**21] = 8

image= skimage.morphology.dilation(image, square(3))

ave_max =0
ave_var =0
for j in range(1031):
    ave_max = ave_max + maximum[extreme[j,0], extreme[j,1]]
    ave_var = ave_var+ variance[extreme[j,0], extreme[j,1]]