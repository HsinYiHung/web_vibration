import os, glob, numpy as np, matplotlib.pyplot as plt, scipy
import imageio
from moviepy.editor import VideoFileClip


### Load the file

freq = 300
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

#Reverse the data to do the backward FFT
#data = data[:, :, ::-1]

### Extract the web index

web_idx = data[:, :, 0]> threshold
res = np.where(web_idx == True)

## Find the signal to noise ratio of the orignal FFT
#def signaltonoise(a, axis=0, ddof=0):
#    a = np.asanyarray(a)
#    m = a.mean(axis)
#    sd = a.std(axis=axis, ddof=ddof)
#    return np.where(sd == 0, 0, m/sd)


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    idx = np.where(a > np.mean(a))
    m_signal = a[idx].mean(axis)
    idx = np.where(a <= np.mean(a))
    m_noise = a[idx].mean(axis)
    #sd = a.std(axis=axis, ddof=ddof)
    return (m_signal/m_noise)

### Apply sliding window to data
fig, ax = plt.subplots(11)
snr = None
snr2_mean = None
snr2_var = None
peak = None
var = None
mean = None
    
for t in range(1000, 10002, 1000):

    dataFFT = np.abs(scipy.fft(data[res[0], res[1], 0:t]))
    ff = np.fft.fftfreq(dataFFT.shape[1], 0.001)
    
    # Get the SNR during 200-400Hz
    idx_i = (np.abs(ff - (freq-100))).argmin()
    idx_e =  (np.abs(ff -  (freq+100))).argmin()
    temp = np.mean(dataFFT, axis = 0)[idx_i:idx_e]
    snr = np.append(snr, signaltonoise(temp))
    
    # Try other methods to calculate SNR: the peak to mean or peak to variance.
    temp2 = list(temp)
    temp2_max = temp.max()
    temp2.remove(temp.max())
    temp2 = np.array(temp2)
    peak = np.append(peak, temp2_max)
    mean = np.append(mean, temp2.mean())
    var = np.append(var, temp2.var())
    snr2_mean = np.append(snr2_mean, temp2_max/temp2.mean())
    snr2_var = np.append(snr2_var, temp2_max/temp2.var())
    
    ax[int(t/1000-1)].plot(ff[ff > 0], np.mean(dataFFT, axis=0)[ff >0], label = 'window =' + str(t))
    ax[int(t/1000-1)].legend(prop={"size":8})




dataFFT = np.abs(scipy.fft(data[res[0], res[1], :]))
#ff = np.fft.fftfreq(dataFFT.shape[1], 0.001)
    
# Get the SNR during 200-400Hz
idx_i = (np.abs(ff - 200)).argmin()
idx_e =  (np.abs(ff - 400)).argmin()
temp = np.mean(dataFFT, axis = 0)[idx_i:idx_e]
snr = np.append(snr, signaltonoise(temp))
temp2 = list(temp)
temp2_max = temp.max()
temp2.remove(temp.max())
temp2 = np.array(temp2)
peak = np.append(peak, temp2_max)
mean = np.append(mean, temp2.mean())
var = np.append(var, temp2.var())
snr2_mean = np.append(snr2_mean, temp2_max/temp2.mean())
snr2_var = np.append(snr2_var, temp2_max/temp2.var())
ax[10].plot(ff[ff > 0], np.mean(dataFFT, axis=0)[ff >0], label = 'window =10002')
ax[10].legend(prop={"size":8}, loc = 'lower left')
plt.savefig('FFT_vibration_web_test_window_400hz.png')


snr = np.delete(snr, 0)
snr2_mean = np.delete(snr2_mean, 0)
snr2_var = np.delete(snr2_var, 0)
peak = np.delete(peak, 0)
var = np.delete(var, 0)
mean = np.delete(mean, 0)


label = [1]*(len(snr)-1)
label.append(0)
xx = [str(i) for i in range(1000, 10002, 1000)]
xx.append('10001')
plt.figure()
plt.scatter(xx, snr, c=label)
plt.figure()
plt.scatter(xx, snr2_mean, c=label)
plt.figure()
plt.scatter(xx, snr2_var, c=label)



plt.figure()
plt.plot(temp)
plt.plot(temp2)