#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 23:39:21 2020

@author: soukhind
"""
from tools.smooth import smooth
import matplotlib.pyplot as plt


fig = plt.figure(figsize = (15,10))
temp = np.diagonal(a1)
plt.plot(smooth(temp,9,window = 'hanning'),label = 'smooth')
plt.plot(np.diagonal(a1),label = 'raw') 
plt.xlim(-0.5,17)
plt.ylim(10,50)
plt.legend()
plt.title("Non linear effects",size = 20)
labels = np.arange(18).astype(str)
plt.xticks(np.arange(18), labels, size = 15)
#%%
x1 = np.zeros((19))
x2 = np.zeros((19))
for lisi in range(1,19):
    d = design.expdesign(1, lisi, lisi, 0.1, 100, [102], lv)
    data = d.tcourse()
    e = design.expanalyse(data, np.array([1]), expdesign = d)
    x1[lisi-1] = e.calc_Fd()
    x2[lisi-1] = e.calc_Fe(ncond = 1)
    print(lisi)

#%%
#%%
from tools import plotdata
plotdata.plotdata(p1,p2,150,3)
#%%

temp = e.roi.reshape(1,e.roi.shape[0])
pca.fit(out)

#%%
def avgHRF(onsets,brain,tr):
    res = 20
    out = np.zeros((len(onsets),res))
    for i in range(0,len(onsets)):
        start = (onsets[i] / tr).astype('int')
        series_A = np.linspace(start - 3,start + 16,res).astype('int')
        if series_A[-1]  >= brain.shape[0]:
            break
        else:
            out[i,:] = brain[series_A]
        
    return out

#out = avgHRF(e.expdesign.onsets_A, e.roi, e.expdesign.loadvolume.tr)
#%%
from scipy.signal import savgol_filter as saf