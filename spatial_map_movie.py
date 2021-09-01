# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 10:04:17 2020

@author: Hsin-Yi
"""

import numpy as np
fname = 'C:/Users/Hsin-Yi/OneDrive - Johns Hopkins/Gordus lab/photron sa5/0220_F2.7_200_400_3_sa5_C001H001S0001.avi'
data = np.load('C:/Users/Hsin-Yi/Documents/GitHub/web_vibration/0220_F2.7_200_400_3_sa5_180-220.npz')

variance= data['variance']
variance_nopeak = data['variance_nopeak']
mean_power = data['mean_power']
maximum = data['maximum']


from loadAnnotations import *
import skimage.draw, numpy as np
from skimage.morphology import square
import os, glob, scipy
import imageio
from moviepy.editor import VideoFileClip
import math
import cv2
from cv2 import VideoWriter_fourcc
import sys
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

N = 256
yellow = np.ones((N, 4))
yellow[:, 0] = np.linspace(1, 255/256, N) # R = 255
yellow[:, 1] = np.linspace(1, 232/256, N) # G = 232
yellow[:, 2] = np.linspace(1, 11/256, N)  # B = 11
yellow_cmp = ListedColormap(yellow)

red = np.ones((N, 4))
red[:, 0] = np.linspace(1, 255/256, N) # R = 255
red[:, 1] = np.linspace(1, 0/256, N) # G = 232
red[:, 2] = np.linspace(1, 65/256, N)  # B = 11
red_cmp = ListedColormap(red)

blue = np.ones((N, 4))
blue[:, 0] = np.linspace(1, 0/256, N) # R = 255
blue[:, 1] = np.linspace(1, 0/256, N) # G = 232
blue[:, 2] = np.linspace(1, 255/256, N)  # B = 11
blue_cmp = ListedColormap(blue)

green = np.ones((N, 4))
green[:, 0] = np.linspace(1, 0/256, N) # R = 255
green[:, 1] = np.linspace(1, 255/256, N) # G = 232
green[:, 2] = np.linspace(1, 0/256, N)  # B = 11
green_cmp = ListedColormap(green)

gray = np.ones((N, 4))
gray[:, 0] = np.linspace(1, 169/256, N) # R = 255
gray[:, 1] = np.linspace(1, 169/256, N) # G = 232
gray[:, 2] = np.linspace(1, 169/256, N)  # B = 11
gray_cmp = ListedColormap(gray)

###Export the result as a video with spider moving and the SNR map
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation
import os, glob, numpy as np, matplotlib.pyplot as plt, scipy
import imageio
from moviepy.editor import VideoFileClip

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
data = np.load(fname + '.npy')
data0 = data[:, :, 0]    
writer = FFMpegWriter(fps=15, metadata=metadata)

#mean_power = variance/mean_power
fig = plt.figure()
ax2=plt.subplot(122)


scale = 200

#mean_power = variance/mean_power
lm = ax2.imshow(maximum[:, :, 0], cmap =red_cmp,interpolation ='nearest', alpha = 1, vmax = scale, vmin = 0)
#ln = ax2.imshow(pltsnr_20[:, :, 0], cmap = blue_cmp,interpolation ='nearest', alpha = 1, vmax = 20, vmin = 0)
#lo = ax2.imshow(pltsnr_30[:, :, 0], cmap = green_cmp,interpolation ='nearest', alpha = 1, vmax = scale, vmin = 0)
#lp = ax2.imshow(pltsnr_60[:, :, 0], cmap = yellow_cmp,interpolation ='nearest', alpha = 1, vmax = scale, vmin = 0)
ax2.imshow(data0, cmap = gray_cmp, alpha = 0.3)

ax1=plt.subplot(121)
im = ax1.imshow(data[:, :, 0], cmap = 'gray')


x=0
c=1
with writer.saving(fig, "test.mp4", 100):
    for i in range(data.shape[2]):
        x+=10
        if x >1000:
            
            lm.set_data(maximum[:, :, c])
            #ln.set_data(pltsnr_20[:, :, c])
            #lo.set_data(pltsnr_30[:, :, c])
            #lp.set_data(pltsnr_60[:, :, c])
            c=c+1
        if c> variance.shape[2]:
            break
        if x> data.shape[2]:
            break
        im.set_data(data[:, :, x])
        
        writer.grab_frame()

#plt.figure()
#plt.imshow(maximum[:, :, 60], cmap =red_cmp,interpolation ='nearest', alpha = 1, vmax = 20000, vmin = 0)
#plt.colorbar()
#plt.show()