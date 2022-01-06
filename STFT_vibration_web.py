import os, glob, numpy as np, matplotlib.pyplot as plt, scipy
import imageio
from moviepy.editor import VideoFileClip
import cv2
from scipy import stats
from scipy import fft
### Load the file

# Load the speaker file
#freq = 300
#fname = glob.glob('web_{}hz*.avi'.format(freq))
#fname = [x for x in fname if not 'spider' in x]
#fname = fname[0]

# Load the spider/flies file
#fname = 'Z:/HsinYi/web_vibration/102121/102121_piezo_5hz_75_182_with_pulses/102121_piezo_5hz_75_182_with_pulses.avi'
#filename = 'Z:/HsinYi/web_vibration/102121/102121_piezo_5hz_75_182_with_pulses/102121_piezo_5hz_75_182_with_pulses.xyt.npy.txt'

fname = 'C:/Users/Hsin-Yi/Documents/GitHub/web_vibration/video/1101_spider_prey.avi'
filename = 'C:/Users/Hsin-Yi/Documents/GitHub/web_vibration/video/1101_spider_prey.xyt.npy.txt'

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
#bottom right portion of the web
#data = data[200:600, 400:800, :]



if fname == 'video/web_whitenoise-006.avi': 
    data = data[:, :, 0:8609]
elif fname == 'video/web_baseline-005.avi': 
    data = data[:, :, 0:8589]
elif fname == 'video/web_flies_1-013.avi':
    data = data[:, :, 0:21494]
    
    
## Load the basline video file
#fbaseline  ='video/web_baseline-005.avi'
#baseline_data = np.load(fbaseline + '.npy')
#baseline_data =baseline_data[:, :, 0:8589]

### Substract the spider/flies/stabalimentum 
kernel = np.ones((5,5),np.uint8)
for i in range(0, data.shape[2], 500):
    erosion = cv2.erode(data[:, :, i:(i+500)],kernel,iterations = 1)
    dilation = cv2.dilate(erosion, kernel,iterations = 1)
    data[:, :, i:(i+500) ] = data[:, :, i:(i+500) ]- dilation
    #data[:, :, i:(i+500) ] = dilation
    


### Extract the web index
threshold = 20
# Use maximum projection as a thresold
#maxprojection = np.amax(data, axis =2)

if fname == 'video/web_whitenoise-006.avi': 
    web_idx = data[:, :, 1800]> threshold
else:
    web_idx = data[:, :, 0]> threshold
#web_idx = maxprojection> threshold
#web_idx = (data[:, :, 0]> 40)& (data[:, :, 0]< 250)
res = np.where(web_idx == True)



### The STFT analysis
f_spec = np.zeros((1000, len(range(1000, data.shape[2], 10))))
z_spec = np.zeros((1000, len(range(1000, data.shape[2], 10))))
v_spec = np.zeros((1000, len(range(1000, data.shape[2], 10))))
#f_spec = np.zeros((1000, len(range(4000, 6000, 10))))

c=0
for t in range(500, (data.shape[2]-500), 10):
#for t in range(4000, 6000, 10):
    dataFFT = np.abs(scipy.fft.fft(data[res[0], res[1], (t-500):(t+500)]))
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
plt.figure()
f_idx =np.where((ff>= 0) & (ff<=50))
img = plt.pcolormesh(t, ff[f_idx], f_spec[f_idx], vmax = 3000)
plt.colorbar()
plt.savefig('test.png')

plt.figure()
img = plt.pcolormesh(t, ff[f_idx], z_spec[f_idx])
plt.colorbar()
plt.figure()
img = plt.pcolormesh(t, ff[f_idx], v_spec[f_idx], vmax = 100)
plt.colorbar()


### Make the power spectrum video
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

data = np.load(fname.replace(".avi", ".xyt") + '.npy')
#data = data[200:600, 400:800, :]


fig = plt.figure()
ax2=plt.subplot(122)
ax2.pcolormesh(t, ff[f_idx], f_spec[f_idx], vmin =0, vmax =3000)
#ax2.pcolormesh(t, ff[f_idx], f_spec[f_idx])



#ln = ax2.axvline(4000, color='red')
ln = ax2.axvline(500, color='red')
x=0
ax1=plt.subplot(121)
im = ax1.imshow(data[:, :, 0], cmap = 'gray')



with writer.saving(fig, "writer_test.mp4", 100):
    for i in range(data.shape[2]):
        x+=10
        if x >500:
            #ln.set_xdata(3000+x)
            ln.set_xdata(x)
            
        if x> data.shape[2]:
            break
        im.set_data(data[:, :, x])
        writer.grab_frame()



#from scipy import signal
#f, t, Zxx = signal.stft(data[res[0], res[1], :], 1000, nperseg=5000)
#plt.pcolormesh(t, f, np.mean(np.abs(Zxx), axis = 0), vmin=0, vmax =2 * np.sqrt(2))
#plt.ylim([100, 500])
#plt.show()

#f, t, Sxx = signal.spectrogram(data[res[0], res[1], :], 1000, return_onesided=False)
#plt.pcolormesh(t, f, np.mean(Sxx, axis =0))
#plt.ylim([0, 500])

