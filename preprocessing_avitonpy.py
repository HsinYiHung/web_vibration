# -*- coding: utf-8 -*-
"""
Created on Sun May 30 17:28:51 2021

@author: Hsin-Yi
"""

import os, glob, numpy as np, matplotlib.pyplot as plt, scipy
import imageio
from moviepy.editor import VideoFileClip
import cv2
from loadAnnotations import *
import skimage.draw, numpy as np
from skimage.morphology import square

### Load the file

# Load the speaker file
#freq = 300
#fname = glob.glob('web_{}hz*.avi'.format(freq))
#fname = [x for x in fname if not 'spider' in x]
#fname = fname[0]

# Load the spider/flies file
#fname = 'video/web_flies_1-013.avi'
fname = 'Z:/HsinYi/web_vibration/110121/1101_spider_piezo_5hz_0_107_with_pulses_2sdelayed_2/1101_spider_piezo_5hz_0_107_with_pulses_2sdelayed_2.avi'
#fname = 'C:/Users/Hsin-Yi/OneDrive - Johns Hopkins/Gordus lab/Chen_camera/white LED/0606_spider001_spider_5_static_C001H001S0001/0606_spider001_spider_5_static_C001H001S0001.avi'

### Convert the video to python data

if os.path.exists(fname.replace(".avi", ".xyt") + '.npy'):
    data = np.load(fname.replace(".avi", ".xyt") + '.npy')
else:
    video = VideoFileClip(fname)
    r = imageio.get_reader(fname)

    data = np.zeros((video.size[0], video.size[1], video.reader.nframes), dtype=np.uint8)
    idx = 0
    for frame in r.iter_data():
        if video.size[0]==video.size[1]:
            data[:, :, idx] = np.mean(frame, axis = 2)
        else:
            data[:, :, idx] = np.mean( np.transpose(frame, (1, 0, 2)), axis = 2)
        idx += 1
    np.save(fname.replace(".avi", ".xyt") + '.npy', data)
    #np.save('C:/Users/Hsin-Yi/Documents/GitHub/web_vibration/video/0619_spider002_spider_prey_top_C001H001S0001.xyt.npy', data)
    #np.save('C:/Users/Hsin-Yi/OneDrive - Johns Hopkins/Gordus lab/Chen_camera/white LED/Team_Spider/0831_spider003_prey.xyt.npy', data)