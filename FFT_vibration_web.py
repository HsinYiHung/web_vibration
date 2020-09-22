import os, glob, numpy as np, matplotlib.pyplot as plt, scipy
import imageio
from moviepy.editor import VideoFileClip
from tqdm import tqdm_notebook as tqdm

### Load the file

freq = 200

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



### Apply FFT to data

#The kernel dies...
dataFFT = np.zeros(data.shape, dtype=np.complex64)
for x in range(data.shape[0]):
    for y in range(data.shape[1]):
        dataFFT[x,y,:] = scipy.fft(data[x,y,:])

