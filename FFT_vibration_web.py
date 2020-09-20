import os, glob, numpy as np, matplotlib.pyplot as plt, scipy
import moviepy
from moviepy.editor import VideoFileClip
from tqdm import tqdm_notebook as tqdm

### Load the file

freq = 400

fname = glob.glob('web_{}hz*.avi'.format(freq))
fname = [x for x in fname if not 'spider' in x]
fname = fname[0]

fnameFFT = fname + '.fft.npy'


### Convert the video to python data

if os.path.exists(fname + '.npy'):
    data = np.load(fname + '.npy')
else:
    video = VideoFileClip(fname)

    data = np.zeros((video.size[0], video.size[1], video.reader.nframes), dtype=np.uint8)
    temp = [frame[0,:,0].max()
             for frame in video.iter_frames()]
    c = 0
    for f in tqdm(r.iter_data()):
        data[:, :, c] = np.mean(f, axis=2)
        c += 1

    np.save(fname + '.npy', data)

dataFFT = np.zeros(data.shape, dtype=np.complex64)
for x in tqdm(range(1024)):
    for y in range(1024):
        dataFFT[x,y,:] = scipy.fft(data[x,y,:])

