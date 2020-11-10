from loadAnnotations import *
import skimage.draw, numpy as np
from skimage.morphology import square
import os, glob, scipy
import imageio
from moviepy.editor import VideoFileClip
import math
import cv2
from cv2 import VideoWriter_fourcc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


freq = 500

filename = glob.glob('web_300hz-007.xyt.npy.txt'.format(freq))
annotations = loadAnnotations(filename[0])

lines = annotations[0][3]
points = annotations[0][1]

webmask = np.full((1024, 1024), False, dtype=np.bool)
for line in lines:
    rr, cc, val = skimage.draw.line_aa(line[0], line[1], line[2], line[3])
    webmask[rr, cc] = True
    
for point in points:
    webmask[point[0], point[1]] = True
webmask_origin = webmask
webmask = skimage.morphology.dilation(webmask, square(3))

fname = glob.glob('web_{}hz*.avi'.format(freq))
fname = [x for x in fname if not 'spider' in x]
fname = fname[0]

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

res = np.where(webmask == True)
res_origin = np.where(webmask_origin==True)

dataFFT_web = np.abs(scipy.fft(data[res[0], res[1], :]))
ff = np.fft.fftfreq(dataFFT_web.shape[1], 0.001)

plt.figure()
plt.plot(ff[ff > 0], np.mean(dataFFT_web, axis=0)[ff > 0])
plt.show()

### Subtract the hub
threshold = 180
hub_idx = data[:, :, 0] > threshold
res = np.where(hub_idx == True)
data[res[0], res[1], :] =0
del hub_idx

res = np.where(webmask == True)
res_origin = np.where(webmask_origin==True)
dataFFT_web_no_hub = np.abs(scipy.fft(data[res[0], res[1], :]))
ff = np.fft.fftfreq(dataFFT_web_no_hub.shape[1], 0.001)

plt.figure()
plt.plot(ff[ff > 0], np.mean(dataFFT_web_no_hub, axis=0)[ff > 0])
plt.show()

diff = np.mean(dataFFT_web, axis=0) - np.mean(dataFFT_web_no_hub, axis=0)
plt.figure()
plt.plot(ff[ff > 0], diff[ff > 0])
plt.show()

#FF =np.zeros((499, 91))
#c=0
#for t in range(1000, 10002, 100):
#    dataFFT_web = np.abs(scipy.fft(data[res[0], res[1], (t-1000):t]))
#    ff = np.fft.fftfreq(dataFFT_web.shape[1], 0.001)
#    FF[:, c] = np.mean(dataFFT_web, axis=0)[ff > 0]
#    c=c+1
    


### Write the video for the spectrum
#FFMpegWriter = manimation.writers['ffmpeg']
#metadata = dict(title='Movie Test', artist='Matplotlib',
#                comment='Movie support!')
#writer = FFMpegWriter(fps=15, metadata=metadata)

#fig = plt.figure()
#l, = plt.plot([], [])
#plt.xlim(0, 500)
#plt.ylim(0, 800)

#with writer.saving(fig, "writer_test.mp4", 91):
#    for i in range(91):
#        l.set_data(ff[ff > 0], FF[:, i])
#        writer.grab_frame()

