#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:38:21 2020

@author: soukhind
"""
 
import loadvolume
from tools import plotfs
import time
lv = loadvolume.loadvolume('Participant_03_rest_run02.nii')

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
paradigm = 'attn'
cue_ratio = 1
start = time.time()
for lisi in np.arange(1,2,1):
    l = 0
    for uisi in np.arange(1,8,1):
        if lisi > uisi:
            l+=1
            continue
        if paradigm:
            arg_map = paradigm + 'map'
        else:
            arg_map = None
        d = design.expdesign(lisi, uisi, 0.1, 100, [2], lv, cue_ratio, 
                             noise = False,nonlinear = True,load = arg_map)
        data = d.tcourse()
        e = design.expanalyse(data, np.array([1, 0]), expdesign = d)
        p1[k,l] = e.calc_Fd()
        p2[k,l] = e.calc_Fe(ncond =2)
        
        
        l += 1
    k += 1
    print(lisi)
print(f'Time: {time.time() - start}')
#x1 = .1
#x2 = .3
#x3 = .5
#x4 = .7
#x5 = .9


#%%

fig1,fig2 = plotfs.plotdata(x3,x5,40,40,normalize = False)
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
        
        