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


idxSS = np.random.randint(0,1023,size=(1000,2),dtype=np.int64)
# Create a 1000x1000 similarity matrix...
mtxDist = np.zeros((idxSS.shape[0] * (idxSS.shape[0]-1) //2,), dtype=np.float64)

c = 0
for i0 in tqdm(range(0, idxSS.shape[0])):
    for i1 in range(i0+1, idxSS.shape[0]):
        a = dataFFT[idxSS[i0,0], idxSS[i0,1], :]
        b = dataFFT[idxSS[i1,0], idxSS[i1,1], :]
        mtxDist[c] = np.linalg.norm(a - b)
        c += 1
        
import fastcluster, time
import scipy.spatial.distance, scipy.cluster.hierarchy
import matplotlib.pyplot as plt

t0 = time.time()
clust = fastcluster.linkage(mtxDist, method='ward', preserve_input=True)
t1 = time.time()
t1 - t0
        