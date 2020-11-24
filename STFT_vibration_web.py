import os, glob, numpy as np, matplotlib.pyplot as plt, scipy
import imageio
from moviepy.editor import VideoFileClip


### Load the file

freq = 400
threshold = 20

fname = glob.glob('web_{}hz*.avi'.format(freq))
fname = [x for x in fname if not 'spider' in x]
fname = fname[0]
fname= 'video/web_flies_plucking-010.avi'
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

web_idx = data[:, :, 0]> threshold
res = np.where(web_idx == True)

f_spec = np.zeros((1000, len(range(1000, data.shape[2], 100))))
c=0
for t in range(1000, data.shape[2], 100):

    dataFFT = np.abs(scipy.fft(data[res[0], res[1], (t-1000):t]))
    f_spec[:,c] = np.mean(np.abs(dataFFT), axis =0)
    c += 1
f_spec = f_spec[1:, :]    
ff = np.fft.fftfreq(dataFFT.shape[1], 0.001)
t= [i for i in range(1000, data.shape[2], 100)]
t = np.array(t) 


#f_idx =np.where((ff>= 350) & (ff<=500))
f_idx =np.where((ff>= 0) & (ff<=100))
img = plt.pcolormesh(t, ff[f_idx], f_spec[f_idx], vmax = 1400)
plt.colorbar()





import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure()
ax2=plt.subplot(122)
ax2.pcolormesh(t, ff[f_idx], f_spec[f_idx], vmax = 1400)


ln = ax2.axvline(1000, color='red')
x=0
ax1=plt.subplot(121)
im = ax1.imshow(data[:, :, 0], cmap = 'gray')


with writer.saving(fig, "writer_test.mp4", 100):
    for i in range(data.shape[2]):
        x+=100
        if x >1000:
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

