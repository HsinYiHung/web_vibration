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

### Imaging sampling frequency
sampling_frequency=1000

### Load files
#freq=500
#filename = glob.glob('web_300hz-007.xyt.npy.txt'.format(freq))

#fname = glob.glob('web_{}hz*.avi'.format(freq))
#fname = [x for x in fname if not 'spider' in x]
#fname = fname[0]
#annotations = loadAnnotations(filename[0])

#fname ='Y:/HsinYi/web_vibration/070121/0701_spider003_web2_prey/0701_spider003_web2_prey.avi'
#filename ='Y:/HsinYi/web_vibration/070121/0701_spider003_web2_prey/0701_spider003_web2_prey.xyt.npy.txt'


fname = 'C:/Users/Hsin-Yi/OneDrive - Johns Hopkins/Gordus lab/Chen_camera/white LED/Team_Spider/0811_spider001_web_piezo_112_144_30hz.avi'
filename = 'C:/Users/Hsin-Yi/OneDrive - Johns Hopkins/Gordus lab/Chen_camera/white LED/Team_Spider/0811_spider001_web_piezo_112_144_30hz.xyt.npy.txt'


annotations = loadAnnotations(filename)

fnameFFT = fname.replace(".avi", ".xyt")  + '.fft.npy'

### Convert the video to python data

if os.path.exists(fname.replace(".avi", ".xyt") + '.npy'):
    data = np.load(fname.replace(".avi", ".xyt") + '.npy')
else:
    video = VideoFileClip(fname)
    r = imageio.get_reader(fname)
    
    data = np.zeros((video.size[0], video.size[1], video.reader.nframes), dtype=np.uint8)
    idx = 0
    for frame in r.iter_data():
        data[:, :, idx] = np.mean(frame, axis = 2)
        idx += 1
    #np.save(fname.replace(".avi", ".xyt") + '.npy', data)

#data = data[:, :, :-1] ###The last frame is missing in the some recordings
    
    
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
    

lines = annotations[0][3]
points = annotations[0][1]
    
webmask = np.full(( data.shape[0],  data.shape[1]), False, dtype=np.bool)
for line in lines:
    rr, cc, val = skimage.draw.line_aa(line[0], line[1], line[2], line[3])

    webmask[rr, cc] = True
    
for point in points:
    webmask[point[0], point[1]] = True
        
#Get the hub
#webmask = ((webmask) & (hub_idx))
    
    
webmask_origin = webmask
webmask = skimage.morphology.dilation(webmask, square(3))
    
res = np.where(webmask == True)
res_origin = np.where(webmask_origin==True )
    


dataFFT_web = np.abs(scipy.fft(data[res[0], res[1], :]))
dataFFT =  np.copy(data)
dataFFT[:] = np.nan
dataFFT[res[0], res[1]] = dataFFT_web
ff = np.fft.fftfreq(dataFFT_web.shape[1], 1/sampling_frequency)

for freq in range(30, 100, 100):

    images_snr =[]
    images_lowfreq =[]
    img = (webmask).astype(float)
    img = img *255
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    alpha = 0.8
    beta = ( 1.0 - alpha );
    AUC = np.zeros((dataFFT.shape[0], dataFFT.shape[1]))
    snr = np.zeros((dataFFT.shape[0], dataFFT.shape[1]))
    pixel_intensity = np.zeros((dataFFT.shape[0], dataFFT.shape[1]))
    maximum = np.zeros((dataFFT.shape[0], dataFFT.shape[1]))
    standard_d = np.zeros((dataFFT.shape[0], dataFFT.shape[1]))
        
    #### This block is the code for averaging fft alone the line
    for j in range(len(res_origin[0])):
        x_idx = res_origin[0][j]
        y_idx = res_origin[1][j]
        means = np.nanmean(np.nanmean(dataFFT[(x_idx-1): (x_idx + 2),
                                                  (y_idx-1): (y_idx + 2), :], axis=0), axis=0)
        if math.isnan(means[0]):
                    #snr[x_idx: (x_idx + step), y_idx: (y_idx + step)] = np.nan
            continue
      
        idx_i = (np.abs(ff - (freq-2))).argmin()
        #idx_i = (np.abs(ff - 4.5798)).argmin()
        if freq == 500:
            idx_e =  (np.abs(ff - (freq))).argmin()
        else:
            idx_e =  (np.abs(ff - (freq+2))).argmin()
            #idx_e = (np.abs(ff - 6.4117)).argmin()

        temp = means[idx_i:idx_e]
        auc_short = sum(temp)
        temp2 = list(temp)
        temp2_max = temp.max()
        temp2.remove(temp.max())
        temp2 = np.array(temp2)
        if np.isnan(temp2_max/temp2.std()):
            continue
        AUC[(x_idx - 1): (x_idx + 2), (y_idx - 1): (y_idx + 2)] =  auc_short
        snr[(x_idx-1): (x_idx + 2),(y_idx-1): (y_idx + 2)] = temp2_max/temp2.std()
        pixel_intensity[(x_idx-1): (x_idx + 2),(y_idx-1): (y_idx + 2)] = np.mean(data[(x_idx-1): (x_idx + 2),(y_idx-1): (y_idx + 2), :])
        maximum[(x_idx-1): (x_idx + 2),(y_idx-1): (y_idx + 2)] = temp2_max
        standard_d[(x_idx-1): (x_idx + 2),(y_idx-1): (y_idx + 2)] = temp2.std()
        
    
        
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
    pixel_intensity = pixel_intensity[res[0], res[1]]
    from scipy import stats
    import matplotlib.pyplot as plt
    #r, p = stats.pearsonr(snr_2,pixel_intensity)
    #plt.figure()
    #plt.scatter(snr_2,pixel_intensity, s=0.1)
    #plt.show()
    #snr_2 = np.delete(snr_2, np.argwhere(pixel_intensity2 > 250))
    #pixel_intensity2 =   np.delete(pixel_intensity2, np.argwhere(pixel_intensity2 > 250))
    
    
    # Filename 
    filename = fname.replace(".avi", "_snr_std_square3_webannotation_")+str(int(freq-2))+'-'+str(int(freq+2))+'hz.jpg'
    print(str(int(freq))+ ' SNR max = ' + str(snr.max()))
    snr_plot = np.copy(snr)
    snr_plot[np.where(snr_plot>10)] =10
    plt.figure()
    plt.imshow(grayImage, cmap='gray') # interpolation='none'
    plt.imshow(snr_plot, cmap = 'hot', alpha = alpha)
    plt.colorbar()
    plt.savefig(filename)

    # Filename 
    filename = fname.replace(".avi", "_auc_square3_webannotation_")+'4-9hz.jpg'
    auc_plot = np.copy(AUC)
    #auc_plot[np.where(auc_plot>1000)] =1000
    plt.figure()
    plt.imshow(grayImage, cmap='gray') # interpolation='none'
    plt.imshow(auc_plot, cmap = 'hot', alpha = alpha)
    plt.colorbar()
    plt.savefig(filename)

    
    ### Using CV2 to plot figure
    #snr_plot = snr_plot/snr_plot.max()*255
    #snr_plot = snr_plot.astype(np.uint8)
    
    #img3 = cv2.applyColorMap(snr_plot, cv2.COLORMAP_HOT)
    #dst3 = cv2.addWeighted( img3, alpha, grayImage, beta, 0.0)
    
    # Filename 
    #filename = '200_snr_std_square3_webannotation' +str(int(freq))+'.jpg'
    # Using cv2.imwrite() method 
    # Saving the image 
    #cv2.imwrite(filename, dst3)
    
    
    ### Plot std
    #plt.figure()
    #plt.imshow(grayImage, cmap='gray') # interpolation='none'
    #plt.imshow(standard_d, cmap = 'hot')
    #plt.colorbar()
    #print(str(int(freq))+ ' std mean = ' + str(np.mean(standard_d)*1024*1024/len(res[0])))
    
    #plt.figure()
    #plt.imshow(maximum, cmap = 'hot')
    #plt.clim(0, 1000)
    #plt.colorbar()
    #print(str(int(freq))+' maximum mean = ' + str(np.mean(maximum)*1024*1024/len(res[0])))
    
    
    #test = np.mean(dataFFT[res[0], res[1], :], axis =0)
    #plt.figure()
    #plt.plot(ff[ff > 0], test[ff > 0])
    #plt.show()