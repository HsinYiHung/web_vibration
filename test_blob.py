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

#freq=500
#filename = glob.glob('web_400hz_with_spider-004.xyt.npy.txt'.format(freq))
filename = 'video/web_flies_plucking-010.xyt.npy.txt'

#fname = glob.glob('web_{}hz*.avi'.format(freq))
#fname = [x for x in fname if not 'spider' in x]
#fname = fname[0]
fname = 'video/web_flies_plucking-010.avi'

fnameFFT = fname + '.fft.npy'

### Convert the video to python data

if os.path.exists(fname + '.npy'):
    data = np.load(fname + '.npy')
else:
    video = VideoFileClip(fname)
    r = imageio.get_reader(fname)
    
    data = np.zeros((video.size[0], video.size[1], video.reader.nframes), dtype=np.uint8)
    idx = 0
    for frame in r.iter_data():
        data[:, :, idx] = np.mean(frame, axis = 2)
        idx += 1
    np.save(fname + '.npy', data)
        
data = data[:, :, :-1]
if fname == 'video/web_whitenoise-006.avi': 
    data = data[:, :, 0:8609]
elif fname == 'video/web_baseline-005.avi': 
    data = data[:, :, 0:8589]

im = data[:, :, 0]

# Filter by Area.
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.filterByColor = False
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
params.minArea = 150

detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)

