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
#fname = 'C:/Users/Hsin-Yi/OneDrive - Johns Hopkins/Gordus lab/Chen_camera/white LED/Team_Spider/0814_spider003_spider_prey.avi'
#filename = 'C:/Users/Hsin-Yi/OneDrive - Johns Hopkins/Gordus lab/Chen_camera/white LED/Team_Spider/0814_spider003_spider_prey.xyt.npy.txt'

#fname = 'C:/Users/Hsin-Yi/Documents/GitHub/web_vibration/video/101821_piezo_5hz_0_107_with_pulses.avi'
#filename = 'C:/Users/Hsin-Yi/Documents/GitHub/web_vibration/video/101821_piezo_5hz_0_107_with_pulses.xyt.npy.txt'
fname = 'Z:/HsinYi/web_vibration/101821/101821_piezo_5hz_75_182_with_pulses/101821_piezo_5hz_75_182_with_pulses.avi'
filename = 'Z:/HsinYi/web_vibration/101821/101821_piezo_5hz_75_182_with_pulses/101821_piezo_control.xyt.npy.txt'

#fname = 'Y:/HsinYi/web_vibration/070121/0701_spider003_web3_control_spider/0701_spider003_web3_control_spider.avi'
#filename = 'Y:/HsinYi/web_vibration/070121/0701_spider003_web3_control_spider/0701_spider003_web3_control_spider.xyt.npy.txt'
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


from skimage.feature import blob_dog, blob_log, blob_doh
data = data[800:1280, 0:500, :]


### Substract the spider/flies/stabalimentum 
#kernel = np.ones((3,3),np.uint8)
#for i in range(0, data.shape[2], 500):
#    erosion = cv2.erode(data[:, :, i:(i+500)],kernel,iterations = 1)
#    dilation = cv2.dilate(erosion, kernel,iterations = 1)
#    #data[:, :, i:(i+500) ] = data[:, :, i:(i+500) ]- dilation
#   data[:, :, i:(i+500) ] =  dilation
    


### Get web by threshold   
threshold = 200
web_idx = data[:, :, 0]> threshold
res = np.where(web_idx == True)



### Apply FFT to data
dataFFT = np.abs(scipy.fft(data[res[0], res[1], :]))

ff = np.fft.fftfreq(dataFFT.shape[1], 0.001)
plt.figure()
plt.plot(ff[ff > 0], np.mean(dataFFT, axis=0)[ff > 0])
plt.savefig(fname.replace('.avi', 'piezo_fft.png'))
plt.figure()
plt.plot(ff[(ff > 0)&(ff<100)], np.mean(dataFFT, axis=0)[(ff > 0)&(ff<100)])
plt.savefig(fname.replace('.avi', 'piezo_2_fft.png'))



from scipy import stats
### The STFT analysis
f_spec = np.zeros((1000, len(range(1000, data.shape[2], 10))))
z_spec = np.zeros((1000, len(range(1000, data.shape[2], 10))))
v_spec = np.zeros((1000, len(range(1000, data.shape[2], 10))))
#f_spec = np.zeros((1000, len(range(4000, 6000, 10))))

c=0
for t in range(500, (data.shape[2]-500), 10):
#for t in range(4000, 6000, 10):
    dataFFT = np.abs(scipy.fft(data[res[0], res[1], (t-500):(t+500)]))
    f_spec[:,c] = np.mean(dataFFT, axis =0)
    z_score = stats.zscore(dataFFT, axis =0)
    z_spec[:,c] = np.mean(np.abs(z_score), axis =0)
    v_spec[:,c] = np.var(dataFFT, axis =0)/np.mean(dataFFT, axis =0)
    #dataFFT_b = np.abs(scipy.fft(baseline_data[res[0], res[1], (t-1000):t]))
    #f_spec[:,c] = np.mean(np.abs(dataFFT), axis =0)/np.mean(np.abs(dataFFT_b), axis =0)
    
    c += 1
f_spec = f_spec[1:, :]
z_spec = z_spec[1:, :]
v_spec = v_spec[1:, :]
ff = np.fft.fftfreq(dataFFT.shape[1], 0.001)
t= [i for i in range(500, (data.shape[2]-500), 10)]
#t= [i for i in range(4000, 6000, 10)]
t = np.array(t) 


### Plot the power spectrum
#f_idx =np.where((ff>= 350) & (ff<=500))
f_idx =np.where((ff>= 0) & (ff<=500))
img = plt.pcolormesh(t, ff[f_idx], f_spec[f_idx])
plt.colorbar()
plt.figure()
img = plt.pcolormesh(t, ff[f_idx], z_spec[f_idx])
plt.colorbar()
plt.figure()
img = plt.pcolormesh(t, ff[f_idx], v_spec[f_idx])
plt.colorbar()
