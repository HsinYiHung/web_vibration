from loadAnnotations import *
import skimage.draw, numpy as np
from skimage.morphology import square
import os, glob, scipy
import imageio
from moviepy.editor import VideoFileClip
import math
import cv2
from cv2 import VideoWriter_fourcc
import sys
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap


### Load the file for web with speaker
#freq=500
#filename = glob.glob('web_400hz_with_spider-004.xyt.npy.txt'.format(freq))
#fname = glob.glob('web_{}hz*.avi'.format(freq))
#fname = [x for x in fname if not 'spider' in x]
#fname = fname[0]


### Load the file for web with spider or flies
#filename = 'video/web_flies_plucking-010.xyt.npy.txt'
#fname = 'video/web_flies_plucking-010.avi'
#fnameFFT = fname + '.fft.npy'

fname = 'C:/Users/Hsin-Yi/OneDrive - Johns Hopkins/Gordus lab/photron sa5/0220_F2.7_200_300_3_sa5_C001H001S0001.avi'
filename = 'C:/Users/Hsin-Yi/OneDrive - Johns Hopkins/Gordus lab/photron sa5/0220_F2.7_200_3_sa5_C001H001S0001.xyt.txt'
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

# Remove the last frame because the last frame is blank 
data = data[:, :, :-1]
# Remove the missing frame in the whitenoise and the baseline video
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

#length = min(baseline_data.shape[2], data.shape[2])    
#baseline_data = baseline_data[:, :, 0:length]
#data = data[:, :, 0:length]  
 
### Extract the hub
#threshold = 180
#hub_idx = data[:, :, 0] > threshold
#res = np.where(hub_idx == True)
#res_null = np.where(hub_idx == False)
#data[res_null[0], res_null[1], :] =0
    
    
### Subtract the hub
#threshold = 180
#hub_idx = data[:, :, 0] > threshold
#res = np.where(hub_idx == True)
#data[res[0], res[1], :] =0
#del hub_idx
    
annotations = loadAnnotations(filename)
    
lines = annotations[0][3]
points = annotations[0][1]
    
webmask = np.full((1024, 1024), False, dtype=np.bool)
for line in lines:
    rr, cc, val = skimage.draw.line_aa(line[0], line[1], line[2], line[3])
    idx1 = np.argwhere(rr>=1024)
    idx2 = np.argwhere(cc>=1024)
    idx = idx1 or idx2
    cc  = np.delete(cc, idx)
    rr  = np.delete(rr, idx)
    webmask[rr, cc] = True
    
for point in points:
    if point[0]>=1024 or point[1]>=1024:
        continue
    webmask[point[0], point[1]] = True
        
#Get the hub
#webmask = ((webmask) & (hub_idx))
    
    
webmask_origin = webmask
webmask = skimage.morphology.dilation(webmask, square(3))
    
res = np.where(webmask == True)
res_origin = np.where(webmask_origin==True )
data0 = data[:, :, 0]    

# due to memory error, I use only 10000 frames
data = data[:, :, 0:10000]

pltsnr_10 =  np.empty((1024, 1024, len(range(1000, data.shape[2], 10))))
#pltsnr_20 =  np.empty((1024, 1024, len(range(1000, data.shape[2], 100))))
#pltsnr_30 =  np.empty((1024, 1024, len(range(1000, data.shape[2], 100))))
#pltsnr_60 =  np.empty((1024, 1024, len(range(1000, data.shape[2], 100))))
# create yellow colormap
N = 256
yellow = np.ones((N, 4))
yellow[:, 0] = np.linspace(1, 255/256, N) # R = 255
yellow[:, 1] = np.linspace(1, 232/256, N) # G = 232
yellow[:, 2] = np.linspace(1, 11/256, N)  # B = 11
yellow_cmp = ListedColormap(yellow)

red = np.ones((N, 4))
red[:, 0] = np.linspace(1, 255/256, N) # R = 255
red[:, 1] = np.linspace(1, 0/256, N) # G = 232
red[:, 2] = np.linspace(1, 65/256, N)  # B = 11
red_cmp = ListedColormap(red)

blue = np.ones((N, 4))
blue[:, 0] = np.linspace(1, 0/256, N) # R = 255
blue[:, 1] = np.linspace(1, 0/256, N) # G = 232
blue[:, 2] = np.linspace(1, 255/256, N)  # B = 11
blue_cmp = ListedColormap(blue)

green = np.ones((N, 4))
green[:, 0] = np.linspace(1, 0/256, N) # R = 255
green[:, 1] = np.linspace(1, 255/256, N) # G = 232
green[:, 2] = np.linspace(1, 0/256, N)  # B = 11
green_cmp = ListedColormap(green)

gray = np.ones((N, 4))
gray[:, 0] = np.linspace(1, 169/256, N) # R = 255
gray[:, 1] = np.linspace(1, 169/256, N) # G = 232
gray[:, 2] = np.linspace(1, 169/256, N)  # B = 11
gray_cmp = ListedColormap(gray)


variance =  np.empty((1024, 1024, len(range(1000, data.shape[2], 10))))
mean_power = np.empty((1024, 1024, len(range(1000, data.shape[2], 10))))
maximum = np.empty((1024, 1024, len(range(1000, data.shape[2], 10))))
variance_nopeak = np.empty((1024, 1024, len(range(1000, data.shape[2], 10))))

c=0
for t in range(500, data.shape[2]-500, 10):
    
    dataFFT_web = np.abs(scipy.fft(data[res[0], res[1], (t-500):(t+500)]))
    #dataFFT_baseline = np.abs(scipy.fft(baseline_data[res[0], res[1], (t-1000):t]))
    
    
    ff = np.fft.fftfreq(dataFFT_web.shape[1], 0.001)
    dataFFT_web = dataFFT_web[:, 0:int(dataFFT_web.shape[1]/2)]
    dataFFT =  np.empty((1024, 1024, dataFFT_web.shape[1]))
    dataFFT[:] = np.nan
    dataFFT[res[0], res[1]] = dataFFT_web
    
    #dataFFT_baseline = dataFFT_baseline[:, 0:int(dataFFT_baseline.shape[1]/2)]
    #dataFFT[res[0], res[1]] = dataFFT_web/dataFFT_baseline

    images_snr =[]
    images_lowfreq =[]
    img = (webmask).astype(float)
    img = img *255
    img = img.astype(np.uint8)
    #grayImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    alpha = 0.8
    beta = ( 1.0 - alpha ); 
    snr = np.zeros((dataFFT.shape[0], dataFFT.shape[1]))
    #pixel_intensity = np.zeros((dataFFT.shape[0], dataFFT.shape[1]))
    #maximum = np.zeros((dataFFT.shape[0], dataFFT.shape[1]))
    #standard_d = np.zeros((dataFFT.shape[0], dataFFT.shape[1]))
    #peak = np.zeros((dataFFT.shape[0], dataFFT.shape[1]))
        
    #### This block is the code for averaging fft alone the line
    for j in range(len(res_origin[0])):
        x_idx = res_origin[0][j]
        y_idx = res_origin[1][j]
        means = np.nanmean(np.nanmean(dataFFT[(x_idx-1): (x_idx + 2),
                                                  (y_idx-1): (y_idx + 2), :], axis=0), axis=0)
        if math.isnan(means[0]) or math.isnan(means[1]):
                    #snr[x_idx: (x_idx + step), y_idx: (y_idx + step)] = np.nan
            continue
      
        idx_i = (np.abs(ff - 180)).argmin()
        idx_e =  (np.abs(ff - 220)).argmin()
        #if freq == 500:
        #    idx_e =  (np.abs(ff - (freq))).argmin()
        #else:
        #    idx_e =  (np.abs(ff - (freq+2))).argmin()
        temp = means[idx_i:idx_e]
        temp2 = list(temp)
        temp2_max = temp.max()
        maximum[(x_idx-1): (x_idx + 2),(y_idx-1): (y_idx + 2), c] = temp.max()
        #peak[(x_idx-1): (x_idx + 2),(y_idx-1): (y_idx + 2)] = np.argmax(temp2)
        variance[(x_idx-1): (x_idx + 2),(y_idx-1): (y_idx + 2), c] = temp.var()
        mean_power[(x_idx-1): (x_idx + 2),(y_idx-1): (y_idx + 2), c] = temp.mean()
        
        temp2.remove(temp.max())
        temp2 = np.array(temp2)
        variance_nopeak[(x_idx-1): (x_idx + 2),(y_idx-1): (y_idx + 2), c] = temp2.var()
        
        idx_i = (np.abs(ff - 450)).argmin()
        idx_e =  (np.abs(ff -490)).argmin()
        temp2 = means[idx_i:idx_e]

        if np.isnan(temp2_max/temp2.std()):
            continue
        snr[(x_idx-1): (x_idx + 2),(y_idx-1): (y_idx + 2)] = temp2_max/temp2.std()
        #pixel_intensity[(x_idx-1): (x_idx + 2),(y_idx-1): (y_idx + 2)] = np.mean(data[(x_idx-1): (x_idx + 2),(y_idx-1): (y_idx + 2), :])
        #maximum[(x_idx-1): (x_idx + 2),(y_idx-1): (y_idx + 2)] = temp2_max
        #standard_d[(x_idx-1): (x_idx + 2),(y_idx-1): (y_idx + 2)] = temp2.std()
        
        

        
        #### This block is the code for calculating the snr and low frequency for dilated images
        #idx_i = (np.abs(ff - (freq-100))).argmin()
        #idx_e =  (np.abs(ff - (freq+100))).argmin()
        #fft = dataFFT_web[:, idx_i:idx_e]
        #fft_max = np.amax(fft, axis =1)
        #m, n = fft.shape
        #temp = np.where(np.arange(n-1) < fft.argmax(axis=1)[:, None], fft[:, :-1], fft[:, 1:])
        #fft_var = np.var(temp, axis =1)
        #index = np.where(fft_var < (1e-10))[0]
        #snr[res[0], res[1]] = fft_max / fft_var
        #snr = np.nan_to_num(snr, 1)
        #snr[np.where(snr>1)] =1
        #low_freq[res[0], res[1]] = np.mean(dataFFT_web[:, 1:1000], axis =1)
    
    ### Plot the snr_plot vs pixel intensity
    snr_2 = snr[res[0], res[1]]
    #pixel_intensity = pixel_intensity[res[0], res[1]]
    from scipy import stats
    import matplotlib.pyplot as plt
    #r, p = stats.pearsonr(snr_2,pixel_intensity)
    #plt.figure()
    #plt.scatter(snr_2,pixel_intensity, s=0.1)
    #plt.show()
    #snr_2 = np.delete(snr_2, np.argwhere(pixel_intensity2 > 250))
    #pixel_intensity2 =   np.delete(pixel_intensity2, np.argwhere(pixel_intensity2 > 250))
    
    #scale = np.max(snr_2[snr_2<10**11])
    scale = 100
    
    # get peak between 10-20 HZ
    #colormap = (peak>0) & (peak <=100)
    pltsnr = np.copy(snr)
    #pltsnr[colormap==False]=0
    pltsnr[pltsnr==0]=np.nan
    pltsnr_10[:, :, c]=np.copy(pltsnr)
    #plt.colorbar()
    # get peak between 20-30 HZ
    #colormap = (peak>100) & (peak <=200)
    #pltsnr = np.copy(snr)
    #pltsnr[colormap==False]=0
    #pltsnr[pltsnr==0]=np.nan
    #pltsnr_20[:, :, c]=np.copy(pltsnr)
    # get peak between 30-40 HZ
    #colormap = (peak>200) & (peak <=300)
    #pltsnr = np.copy(snr)
    #pltsnr[colormap==False]=0
    #pltsnr[pltsnr==0]=np.nan
    #pltsnr_30[:, :, c]=np.copy(pltsnr)
    # get peak between 60-70 HZ
    #colormap = (peak>=500) & (peak <=600)
    #pltsnr = np.copy(snr)
    #pltsnr[colormap==False]=0
    #pltsnr[pltsnr==0]=np.nan
    #pltsnr_60[:, :, c]=np.copy(pltsnr)
    
    c=c+1

np.savez('0220_F2.7_200_300_3_sa5_180-220', maximum= maximum, mean_power= mean_power, variance = variance, variance_nopeak= variance_nopeak, pltsnr = pltsnr_10, fname= fname)
