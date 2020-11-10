from loadAnnotations import *
import skimage.draw, numpy as np
from skimage.morphology import square
import os, glob, scipy
import imageio
from moviepy.editor import VideoFileClip
import math
import cv2
from cv2 import VideoWriter_fourcc


freq = 200

filename = glob.glob('web_300hz-007.xyt.npy.txt'.format(freq))
annotations = loadAnnotations(filename[0])

lines = annotations[0][3]
points = annotations[0][1]

webmask = np.full((1024, 1024), False, dtype=np.bool)
for line in lines:
    rr, cc, val = skimage.draw.line_aa(line[0], line[1], line[2], line[3])
    webmask[rr, cc] = True
    
for point in points:
    webmask[point[0], point[1]] = True
webmask_origin = webmask
#webmask = skimage.morphology.dilation(webmask, square(11))

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
### Extract the web index
res = np.where(webmask == True)
res_origin = np.where(webmask_origin==True)


images_snr =[]
images_lowfreq =[]
img = (webmask).astype(float)
img = img *255
img = img.astype(np.uint8)
grayImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
alpha = 0.8
beta = ( 1.0 - alpha ); 
SNR = np.zeros((data.shape[0], data.shape[1], 91))
Low_freq = np.zeros((data.shape[0], data.shape[1], 91))
    
#### This block is the code for averaging fft alone the line
#for j in range(len(res_origin[0])):
c=0

for t in range(1000, 10002, 100):
    dataFFT_web = np.abs(scipy.fft(data[res[0], res[1], (t-1000):t]))
    dataFFT =  np.empty((1024, 1024, 1000))
    dataFFT[:] = np.nan
    dataFFT[res[0], res[1]] = dataFFT_web
    ff = np.fft.fftfreq(dataFFT_web.shape[1], 0.001)
    snr = np.zeros((data.shape[0], data.shape[1]))
    low_freq = np.zeros((data.shape[0], data.shape[1]))
    
    for line in lines:
        rr, cc, val = skimage.draw.line_aa(line[0], line[1], line[2], line[3])
        means = np.nanmean(dataFFT[rr,cc,  :], axis=0)

        #x_idx = res_origin[0][j]
        #y_idx = res_origin[1][j]
        #means = np.nanmean(np.nanmean(dataFFT[(x_idx-5): (x_idx + 6),
        #                                          (y_idx-5): (y_idx + 6), :], axis=0), axis=0)
        if math.isnan(means[0]):
                    #snr[x_idx: (x_idx + step), y_idx: (y_idx + step)] = np.nan
            continue
  
        idx_i = (np.abs(ff - (freq-100))).argmin()
        idx_e =  (np.abs(ff - (freq+100))).argmin()
        temp = means[idx_i:idx_e]
        temp2 = list(temp)
        temp2_max = temp.max()
        temp2.remove(temp.max())
        temp2 = np.array(temp2)
        low_freq[rr, cc] = np.mean(means[1:100])
        snr[rr, cc] = temp2_max/temp2.mean()
        
    SNR[:, :, c] = snr
    Low_freq[:, :, c] = low_freq
    c = c+1
    
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
    

print('SNR max = ' + str(SNR.max()))
SNR = SNR/SNR.max()*255
SNR = SNR.astype(np.uint8)

print('Low freq max = ' + str(Low_freq.max()))
Low_freq = Low_freq/Low_freq.max()*255
Low_freq = Low_freq.astype(np.uint8)

#img3 = cv2.applyColorMap(SNR, cv2.COLORMAP_HOT)
#dst3 = cv2.addWeighted( img3, alpha, grayImage, beta, 0.0)

# Filename 
#filename = '500_snr_segment_webannotation.jpg'
  
# Using cv2.imwrite() method 
# Saving the image 
#cv2.imwrite(filename, dst3)
for j in range(0, 91):
    img2 = cv2.applyColorMap(Low_freq[:, :, j], cv2.COLORMAP_HOT)
    dst = cv2.addWeighted( img2, alpha, grayImage, beta, 0.0)
    images_lowfreq.append(dst)

    
    
for j in range(0, 91):   
    img3 = cv2.applyColorMap(SNR[:, :, j], cv2.COLORMAP_HOT)
    dst3 = cv2.addWeighted( img3, alpha, grayImage, beta, 0.0)
    images_snr.append(dst3)

fourcc = VideoWriter_fourcc(*'MP42')
video=cv2.VideoWriter('200_lowfreq_segment_webannotation.avi',fourcc,1,(1024,1024))
video3=cv2.VideoWriter('200_snr_segment_webannotation.avi',fourcc,1,(1024,1024))

for j in range(0,len(images_snr)):
    video.write(images_lowfreq[j])
    video3.write(images_snr[j])

cv2.destroyAllWindows()
video.release()
video3.release()