import numpy as np
from scipy import stats

 
 
def computeMI(x, y, x_bin, y_bin):
    sum_mi = 0.0
    hist_x, bin_edges_x = np.histogram(x, bins = x_bin)
    Px = hist_x/np.sum(hist_x)
    hist_y, bin_edges_y = np.histogram(y, bins = y_bin)
    Py = hist_y/np.sum(hist_y)
    
    jointProb, edges = np.histogramdd([x,y], bins = (x_bin, y_bin))
    jointProb /=jointProb.sum()
    
    
    for i in range(x_bin):
        if Px[i] ==0.:
            continue

        pxy = jointProb[i,:]
        t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
        pxy=pxy[Py>0.]
        sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    return sum_mi

def freedman_diaconis(data, returnas="width"):
    """
    Use Freedman Diaconis rule to compute optimal histogram bin width. 
    ``returnas`` can be one of "width" or "bins", indicating whether
    the bin width or number of bins should be returned respectively. 


    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.

    returnas: {"width", "bins"}
        If "width", return the estimated width for each histogram bin. 
        If "bins", return the number of bins suggested by rule.
    """
    data = np.asarray(data, dtype=np.float_)
    IQR  = stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
    N    = data.size
    bw   = (2 * IQR) / np.power(N, 1/3)

    if returnas=="width":
        result = bw
    else:
        datmin, datmax = data.min(), data.max()
        datrng = datmax - datmin
        result = int((datrng / bw) + 1)
    return(result)

f_spec_web = np.load('result/web_flies_1-013_STFT_web.npz')
f_spec_web = f_spec_web['f_spec']
f_spec_flies = np.load('result/web_flies_1-013_STFT_flies2_nostabilimentum.npz')
f_spec_flies = f_spec_flies['f_spec']

mi_matrix = np.zeros((500,500))
correlation =  np.zeros((500,500))
for i in range(0, 500):
    for j in range(0, 500):
        x= f_spec_flies[i, :]
        y= f_spec_web[j, :]
        x_bin = freedman_diaconis(data=x, returnas="bins")
        y_bin = freedman_diaconis(data=y, returnas="bins")
        #np.random.shuffle(x)
        #np.random.shuffle(y)
        #N_bin = np.max([x_bin, y_bin])
        mi = computeMI(x, y, x_bin, y_bin)
        mi_matrix[i,j] = mi
        r=stats.pearsonr(x, y)[0]
        correlation[i,j] = r
        


import matplotlib.pyplot as plt
plt.imshow(mi_matrix[0:100, 0:100], cmap = 'jet')
plt.clim(0, 2)
plt.colorbar()    

plt.figure()
plt.imshow(correlation[0:100, 0:100], cmap = 'jet')
plt.clim(-1, 1)
plt.colorbar()

temp2 = np.sum(mi_matrix, axis =0)
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(temp[0:100], marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylabel('original signal', color=color)
ax2 = ax1.twinx() 
color = 'tab:blue'
ax2.plot(temp2[0:100], marker='o', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('shuffling signal', color=color)