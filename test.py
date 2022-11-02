#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:38:21 2020

@author: soukhind
"""
 
import loadvolume
from tools import plotfs
import time
lv = loadvolume.loadvolume('/Users/soukhind/Desktop/FMRIsim/Full_Data/Corr_MVPA_Data_dataspace/Participant_03_rest_run01.nii')

lv.loaddata()
lv.loadmask()
lv.generate_noise()
lv.generate_region()
maxp1 = 0
maxp2 = 0
#%%

import design
import numpy as np
from scipy.integrate import quad,simps
from scipy.integrate import trapz
import matplotlib.pyplot as plt
import importlib as imp
import matplotlib.pyplot as plt
#%matplotlib qt
p1 = np.zeros((20,20))
p2 = np.zeros((20,20))


k = 0
paradigm = ''
cue_ratio = 1
dist = 'uniform'
start = time.time()
store = 1
for lisi in np.arange(1,21,1):
    l = 0
    for uisi in np.arange(1,21,1):
        if lisi > uisi:
            l += 1
            continue
        if paradigm:
            arg_map = paradigm + 'map'
        else:
            arg_map = None
            
        d = design.expdesign(lisi, uisi, 0.1, 100, [2], lv, dist,30,
                             cue_ratio,noise = False,nonlinear = True,load = arg_map)
                
        data = d.tcourse()
        e = design.expanalyse(data, np.array([1, 0]), expdesign = d)
        p1[k,l] = e.calc_Fd()
        p2[k,l] = e.calc_Fe(ncond =2)
        if store:
            if lisi == 2 and uisi == 3:
                e1 = e.roi
                t1 = e.design[:,0] + e.design[:,1]
            if lisi == 5 and uisi == 9:
                e2 = e.roi
                t2 = e.design[:,0] + e.design[:,1]
            if lisi == 2 and uisi == 19:
                e3 = e.roi
                t3 = e.design[:,0] + e.design[:,1]
            if lisi == 18 and uisi == 19:
                e4 = e.roi
                t4 = e.design[:,0] + e.design[:,1]
            if lisi == 2 and uisi == 10:
                e5 = e.roi
                t5 = e.design[:,0] + e.design[:,1]

        l += 1
    k += 1
    print(lisi)
print(f'Time: {time.time() - start}')



 #%%

fig1,fig2 = plotfs.plotdata(p1,p2,40,40,normalize = True)
#fig1.savefig("Figures/Doubletrial/Final/dt_det_trans_nonlin_" + str(cue_ratio) + "_" + paradigm + '.png' , dpi = 600,bbox_inches = 'tight')
#fig2.savefig("Figures/Doubletrial/Final/dt_heff_trans_nonlin_" + str(cue_ratio) + "_" + '.png',dpi = 600,bbox_inches = 'tight')
#plotdata.plotdata(c1,c3,4500,4500)
#%%
from numpy.fft import fft
from scipy.signal import periodogram
from tools._dghrf import _dghrf,hrf2
hrf = np.asarray(_dghrf())
y = np.asarray(hrf2(np.arange(30)))
zero = np.zeros(30)
x = np.concatenate((zero,y,zero))
#plt.plot(hrf)
#plt.plot(fft(x))
p = periodogram(x,1)
plt.plot(p[1])
labels = np.round(p[0],2).astype(str)
plt.xticks(np.arange(30),labels)
#%%
plt.figure()
plt.plot(d.temp) 
        
        
