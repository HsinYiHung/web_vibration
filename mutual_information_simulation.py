# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:50:28 2020

@author: Hsin-Yi
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
 
 
def computeMI(x, y, N_bin):
    sum_mi = 0.0
    hist_x, bin_edges_x = np.histogram(x, bins = N_bin)
    Px = hist_x/np.sum(hist_x)
    hist_y, bin_edges_y = np.histogram(y, bins = N_bin)
    Py = hist_y/np.sum(hist_y)
    
    jointProb, edges = np.histogramdd([x,y], N_bin)
    jointProb /=jointProb.sum()
    
    
    for i in range(N_bin):
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


### Simulation 1: sine wave
t=np.array(range(1,1101))*0.02*np.pi
x1=np.sin(t)


mi_matrix = np.zeros((100,1))
for i in range(100):
    x=x1[101:1100]
    y=x1[101-i:1100-i]

    mi_matrix[i,0] = computeMI(x, y, N_bin=4)


plt.figure()
plt.plot(mi_matrix)

### Simulation 2:The tent map
mi_matrix = np.zeros((25,1))
c=0
for z in np.linspace(0.002, 0.052, 25):
    A= np.zeros((100000, 100))
    B=np.zeros((1,100))
    C = np.zeros((1,100))
    B[0, :] = np.random.rand(1,100)
    for w in range(1, 100000):
        for n in range(1,100):
            if (n-1)==0:
                b=99
            else:
                b=n-1
            C[0,n]= z*B[0,b]+(1-z)*B[0,n]
            if C[0, n] <0.5:
                C[0,n] = C[0, n]*2
            else:
                C[0,n] = 2-C[0, n]*2
        for n in range(1,100):
            B[0,n] = C[0,n]
        C = np.zeros((1,100))
    for m in range(1,100):
        if (m-1)==0:
            b=99
        else:
            b=m-1
        A[0,m] = z*B[0,b]+(1-z)*B[0,m]
        if A[0, m] <0.5:
            A[0,m] = A[0, m]*2
        else:
            A[0,m] = 2-A[0, m]*2
    for n in range(1,100000):
        for m in range(100):
            if (m-1)==0:
                b=99
            else:
                b=m-1
            A[n,m]= z*A[n-1, b]+(1-z)*A[n-1, m]
            if A[n,m] <0.5:
                A[n, m] = A[n,m] *2
            else:
                A[n, m] = 2-A[n,m] *2
    x=A[:,3]
    y=A[:,2]
    x_bin = freedman_diaconis(data=x, returnas="bins")
    y_bin = freedman_diaconis(data=y, returnas="bins")
    N_bin = np.max([x_bin, y_bin])
    mi_matrix[c,0] = computeMI(x, y, N_bin)
    c=c+1
    
plt.figure()
plt.plot(mi_matrix)