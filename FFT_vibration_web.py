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
    

### Extract the web index

web_idx = data[:, :, 0]> threshold
res = np.where(web_idx == True)


### Apply FFT to data
dataFFT = np.abs(scipy.fft(data[res[0], res[1], :]))

ff = np.fft.fftfreq(10002, 0.001)
plt.figure()
plt.plot(ff[ff > 0], np.mean(dataFFT, axis=0)[ff > 0])
plt.savefig(fname.replace('.avi', '_fft.png'))

