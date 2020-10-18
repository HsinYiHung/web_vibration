import os, glob, numpy as np, scipy
import imageio
from moviepy.editor import VideoFileClip
import math
import cv2
from cv2 import VideoWriter_fourcc
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

#snr = np.zeros((data.shape[0], data.shape[1]))
web_idx = data[:, :, 0]> threshold
res = np.where(web_idx == True)
#dataFFT_web = np.abs(scipy.fft(data[res[0], res[1], :]))

step =16
images =[]
img = (data[:, :, 0]>20).astype(float)
img = img *255
img = img.astype(np.uint8)
grayImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
alpha = 0.8
beta = ( 1.0 - alpha );

for t in range(1000, 10002, 100):
    dataFFT_web = np.abs(scipy.fft(data[res[0], res[1], (t-1000):t]))
    dataFFT =  np.empty((1024, 1024, 1000))
    dataFFT[:] = np.nan
    dataFFT[res[0], res[1]] = dataFFT_web
    ff = np.fft.fftfreq(dataFFT_web.shape[1], 0.001)
    snr = np.zeros((data.shape[0], data.shape[1]))
    for x_idx in range(0, 1025, step):
        for y_idx in range(0, 1025, step):
            means = np.nanmean(np.nanmean(dataFFT[x_idx: (x_idx + step),
                                              y_idx: (y_idx + step), :], axis=0), axis=0)
            if math.isnan(means[0]):
                #snr[x_idx: (x_idx + step), y_idx: (y_idx + step)] = np.nan
                continue
            idx_i = (np.abs(ff - (freq-100))).argmin()
            idx_e =  (np.abs(ff - (freq+100))).argmin()
            temp = means[idx_i:idx_e]
            temp2 = list(temp)
            temp2_max = temp.max()
            temp2.remove(temp.max())
            temp2 = np.array(temp2)
            snr[x_idx: (x_idx + step), y_idx: (y_idx + step)] = temp2_max/temp2.mean()
    SNR = snr/snr.max()*255
    SNR = SNR.astype(np.uint8)
    img2 = cv2.applyColorMap(SNR, cv2.COLORMAP_HOT)
    dst = cv2.addWeighted( img2, alpha, grayImage, beta, 0.0)
    
    images.append(dst)




fourcc = VideoWriter_fourcc(*'MP42')
video=cv2.VideoWriter('300_snr_threshold20_grid16.avi',fourcc,1,(1024,1024))

for j in range(0,len(images)):
    video.write(images[j])

cv2.destroyAllWindows()
video.release()