
import os, glob, numpy as np, matplotlib.pyplot as plt, scipy
import imageio
from moviepy.editor import VideoFileClip


### Load the file

freq = 500
threshold = 20 #luminance threshold for a slik

fname = glob.glob('web_{}hz*.avi'.format(freq))
fname = [x for x in fname if not 'spider' in x]
fname = fname[0]

fnameFFT = fname + '.fft.npy'


### Convert the video to python data

if os.path.exists(os.path.exists(fname.replace(".avi", ".xyt") + '.npy')):
    data = np.load(os.path.exists(fname.replace(".avi", ".xyt") + '.npy'))
else:
    video = VideoFileClip(fname)
    r = imageio.get_reader(fname)

    data = np.zeros((video.size[0], video.size[1], video.reader.nframes), dtype=np.uint8)
    idx = 0
    for frame in r.iter_data():
        data[:, :, idx] = np.mean(frame, axis = 2)
        idx += 1
    np.save(os.path.exists(fname.replace(".avi", ".xyt") + '.npy'), data)
    
data2 = data[:, :, :-1] ###The last frame is missing in the some recordings
data = data2

### Extract the web index


snr = np.zeros((data.shape[0], data.shape[1]))

for divide in range(int(np.log2(data.shape[0]))):
    
    # Space index: Only extract parts of the web
    space = int(data.shape[0]/2)
    space_idx = np.zeros((data.shape[0], data.shape[1]), dtype=bool) 
    # Web index: silk above the luminance threshold
    web_idx = data[:, :, 0]> threshold
    for i in range(0, 2):
        for j in range(0,2):

            space_idx[(space *i) :(space*(i+1)), (space *j) :(space*(j+1))] = True
            res = np.where((web_idx == True) & (space_idx==True))
            dataFFT = np.abs(scipy.fft(data[res[0], res[1], :]))
            if len(dataFFT)==0:
                snr[(space *i) :(space*(i+1)), (space *j) :(space*(j+1))] = 0
                continue
            ff = np.fft.fftfreq(dataFFT.shape[1], 0.001)
            idx_i = (np.abs(ff - 200)).argmin()
            idx_e =  (np.abs(ff - 400)).argmin()
            temp = np.mean(dataFFT, axis = 0)[idx_i:idx_e]
            temp2 = list(temp)
            temp2_max = temp.max()
            temp2.remove(temp.max())
            temp2 = np.array(temp2)
            snr[(space *i) :(space*(i+1)), (space *j) :(space*(j+1))] = temp2_max/temp2.mean()

    data = data[0:space, 0:space, :]
    if divide ==0 :
        data = np.rot90(np.rot90(data))
snr[0:512, 0:512] = np.rot90(np.rot90(snr[0:512, 0:512]))    


data = data2[512:1024, 0:512, :]
data = np.rot90(data)
for divide in range(1, int(np.log2(data.shape[0]))):
    space = int(data.shape[0]/2)
    space_idx = np.zeros((data.shape[0], data.shape[1]), dtype=bool) 
    web_idx = data[:, :, 0]> threshold
    for i in range(0, 2):
        for j in range(0,2):

            space_idx[(space *i) :(space*(i+1)), (space *j) :(space*(j+1))] = True
            res = np.where((web_idx == True) & (space_idx==True))
            dataFFT = np.abs(scipy.fft(data[res[0], res[1], :]))
            if len(dataFFT)==0:
                snr[(space *i) :(space*(i+1)), (space *j) :(space*(j+1))] = 0
                continue
            ff = np.fft.fftfreq(dataFFT.shape[1], 0.001)
            idx_i = (np.abs(ff - 200)).argmin()
            idx_e =  (np.abs(ff - 400)).argmin()
            temp = np.mean(dataFFT, axis = 0)[idx_i:idx_e]
            temp2 = list(temp)
            temp2_max = temp.max()
            temp2.remove(temp.max())
            temp2 = np.array(temp2)
            snr[(512+space *i) :(512+space*(i+1)), (space *j) :(space*(j+1))] = temp2_max/temp2.mean()
    data = data[0:space, 0:space, :]
snr[512:1024, 0:512] = np.rot90(np.rot90(np.rot90(snr[512:1024, 0:512])))    
 


data = data2[512:1024, 512:1024, :]
for divide in range(1, int(np.log2(data.shape[0]))):
    space = int(data.shape[0]/2)
    space_idx = np.zeros((data.shape[0], data.shape[1]), dtype=bool) 
    web_idx = data[:, :, 0]> threshold
    for i in range(0, 2):
        for j in range(0,2):

            space_idx[(space *i) :(space*(i+1)), (space *j) :(space*(j+1))] = True
            res = np.where((web_idx == True) & (space_idx==True))
            dataFFT = np.abs(scipy.fft(data[res[0], res[1], :]))
            if len(dataFFT)==0:
                snr[(space *i) :(space*(i+1)), (space *j) :(space*(j+1))] = 0
                continue
            ff = np.fft.fftfreq(dataFFT.shape[1], 0.001)
            idx_i = (np.abs(ff - 200)).argmin()
            idx_e =  (np.abs(ff - 400)).argmin()
            temp = np.mean(dataFFT, axis = 0)[idx_i:idx_e]
            temp2 = list(temp)
            temp2_max = temp.max()
            temp2.remove(temp.max())
            temp2 = np.array(temp2)
            snr[(512+space *i) :(512+space*(i+1)), (512+space *j) :(512+space*(j+1))] = temp2_max/temp2.mean()
    data = data[0:space, 0:space, :]


data = data2[0:512,512:1024, :]
data = np.rot90(np.rot90(np.rot90(data)))
for divide in range(1, int(np.log2(data.shape[0]))):
    space = int(data.shape[0]/2)
    space_idx = np.zeros((data.shape[0], data.shape[1]), dtype=bool) 
    web_idx = data[:, :, 0]> threshold
    for i in range(0, 2):
        for j in range(0,2):

            space_idx[(space *i) :(space*(i+1)), (space *j) :(space*(j+1))] = True
            res = np.where((web_idx == True) & (space_idx==True))
            dataFFT = np.abs(scipy.fft(data[res[0], res[1], :]))
            if len(dataFFT)==0:
                snr[(space *i) :(space*(i+1)), (space *j) :(space*(j+1))] = 0
                continue
            ff = np.fft.fftfreq(dataFFT.shape[1], 0.001)
            idx_i = (np.abs(ff - 200)).argmin()
            idx_e =  (np.abs(ff - 400)).argmin()
            temp = np.mean(dataFFT, axis = 0)[idx_i:idx_e]
            temp2 = list(temp)
            temp2_max = temp.max()
            temp2.remove(temp.max())
            temp2 = np.array(temp2)
            snr[(space *i) :(space*(i+1)), (512+space *j) :(512+space*(j+1))] = temp2_max/temp2.mean()
    data = data[0:space, 0:space, :]
snr[ 0:512, 512:1024] = np.rot90(snr[ 0:512, 512:1024])
 



plt.imshow(data2[:, :, 0]>20, cmap='gray') # interpolation='none'
plt.imshow(snr, cmap='hot', alpha=0.9)
plt.colorbar()
