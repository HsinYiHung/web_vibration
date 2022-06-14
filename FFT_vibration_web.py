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
#fname = 'C:/Users/Hsin-Yi/OneDrive - Johns Hopkins/Gordus lab/Chen_camera/white LED/0606_spider001_spider_prey_C001H001S0001/0606_spider001_spider_prey_C001H001S0001.avi'
#filename = 'C:/Users/Hsin-Yi/OneDrive - Johns Hopkins/Gordus lab/Chen_camera/white LED/0606_spider001_spider_prey_C001H001S0001/0606_spider001_spider_prey_C001H001S0001.xyt.npy.txt'

fname = 'C:/Users/Hsin-Yi/Documents/GitHub/web_vibration/video/1101_spider_prey.avi'
filename = 'C:/Users/Hsin-Yi/Documents/GitHub/web_vibration/video/1101_spider_prey.xyt.npy.txt'
#fname = 'Z:/HsinYi/web_vibration/070121/0701_spider003_web2_prey/0701_spider003_web2_prey.avi'
#filename = 'Z:/HsinYi/web_vibration/070121/0701_spider003_web2_prey/0701_spider003_web2_prey.xyt.npy.txt'

#fname = 'Y:/HsinYi/web_vibration/070121/0701_spider003_web3_control_spider/0701_spider003_web3_control_spider.avi'
#filename = 'Y:/HbfsinYi/web_vibration/070121/0701_spider003_web3_control_spider/0701_spider003_web3_control_spider.xyt.npy.txt'
fnameFFT = fname + '.fft.npy'


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

### Extract the web index

data = data[:, :, :-1]
#data = data[420:520, 550:650, :]
#data = data[480:580,550:650, :]


if fname == 'video/web_whitenoise-006.avi': 
    data = data[:, :, 0:8609]
elif fname == 'video/web_baseline-005.avi': 
    data = data[:, :, 0:8589]
elif fname == 'video/web_flies_1-013.avi':
    data = data[:, :, 0:21494]


### Substract the spider/flies/stabalimentum 
#kernel = np.ones((3,3),np.uint8)
#for i in range(0, data.shape[2], 500):
#    erosion = cv2.erode(data[:, :, i:(i+500)],kernel,iterations = 1)
#    dilation = cv2.dilate(erosion, kernel,iterations = 1)
#    #data[:, :, i:(i+500) ] = data[:, :, i:(i+500) ]- dilation
#   data[:, :, i:(i+500) ] =  dilation
    


### Get web by threshold   
#threshold = 20
#web_idx = data[:, :, 0]> threshold
#res = np.where(web_idx == True)

### Get web by annotaation
annotations = loadAnnotations(filename)

lines = annotations[0][3]
points = annotations[0][1]

webmask = np.full((np.size(data,0), np.size(data,1)), False, dtype=np.bool)
for line in lines:
    rr, cc, val = skimage.draw.line_aa(line[0], line[1], line[2], line[3])
    #idx1 = np.argwhere(rr>=1024)
    #idx2 = np.argwhere(cc>=1024)
    #if np.size(idx2)==0:
    #    idx = idx1
    #else:
    #    idx = idx1 or idx2
    #cc  = np.delete(cc, idx)
    #rr  = np.delete(rr, idx)
    webmask[rr, cc] = True

for point in points:
    if point[0]>=1024 or point[1]>=1024:
        continue
    webmask[point[0], point[1]] = True

webmask = skimage.morphology.dilation(webmask, square(3))
#webmask[0:800, :]= False
#webmask[800:1280, 0:500]= False
res = np.where(webmask == True)



### Apply FFT to data
dataFFT = np.abs(scipy.fft.fft(data[res[0], res[1], :]))

ff = np.fft.fftfreq(dataFFT.shape[1], 0.001)
plt.figure()
plt.plot(ff[ff > 0], np.mean(dataFFT, axis=0)[ff > 0])
plt.savefig(fname.replace('.avi', '_fft.png'))
plt.figure()
plt.plot(ff[(ff > 0)&(ff<100)], np.mean(dataFFT, axis=0)[(ff > 0)&(ff<100)])
plt.savefig(fname.replace('.avi', '_2_fft.png'))
plt.figure()
plt.plot(ff[(ff > 1)&(ff<100)], np.mean(dataFFT, axis=0)[(ff > 1)&(ff<100)])
plt.savefig(fname.replace('.avi', '_3_fft.png'))

np.savez(fname.replace('.avi', '_fft'), ff=ff, dataFFT = np.abs(scipy.fft.fft(data[res[0], res[1], :])))