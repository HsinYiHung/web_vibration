
import os, glob, numpy as np, matplotlib.pyplot as plt, scipy
import imageio
from moviepy.editor import VideoFileClip
import math
### Load the file

freq = 200
threshold = 20

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
### Extract the web index

snr = np.zeros((data.shape[0], data.shape[1]))
web_idx = data[:, :, 0]> threshold
res = np.where(web_idx == True)
dataFFT_web = np.abs(scipy.fft(data[res[0], res[1], :]))

dataFFT =  np.empty((1024, 1024, 10001))
dataFFT[:] = np.nan
dataFFT[res[0], res[1]] = dataFFT_web
ff = np.fft.fftfreq(dataFFT_web.shape[1], 0.001)

step =16

for x_idx in range(0, 1025, step):
    for y_idx in range(0, 1025, step):
        means = np.nanmean(np.nanmean(dataFFT[x_idx: (x_idx + step),
                                              y_idx: (y_idx + step), :], axis=0), axis=0)
        if math.isnan(means[0]):
            snr[x_idx: (x_idx + step), y_idx: (y_idx + step)] = np.nan
            continue
        idx_i = (np.abs(ff - (freq-100))).argmin()
        idx_e =  (np.abs(ff - (freq+100))).argmin()
        temp = means[idx_i:idx_e]
        temp2 = list(temp)
        temp2_max = temp.max()
        temp2.remove(temp.max())
        temp2 = np.array(temp2)
        snr[x_idx: (x_idx + step), y_idx: (y_idx + step)] = temp2_max/temp2.var()






plt.imshow(data[:, :, 0]>20, cmap='gray') # interpolation='none'
plt.imshow(snr, cmap='hot', alpha=0.7)
plt.colorbar()
