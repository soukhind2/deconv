#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:38:21 2020

@author: soukhind
"""

import loadvolume

lv = loadvolume.loadvolume('Participant_03_rest_run02.nii')

lv.loaddata()
lv.loadmask()
lv.generate_noise()
lv.generate_region()

#%%

import design
import numpy as np
from scipy.integrate import quad,simps
from scipy.integrate import trapz
import matplotlib.pyplot as plt
import importlib as imp
import matplotlib.pyplot as plt
p1 = np.zeros((18,18))
p2 = np.zeros((18,18))
a1 = np.zeros((18,18))


k = 0

for lisi in np.arange(1,19,1):
    l = 0
    for uisi in np.arange(1,19,1):
        if lisi > uisi:
            l+=1
            continue
        d = design.expdesign(1, lisi, uisi, 0.1, 100, [2], lv)
        data = d.tcourse()
        e = design.expanalyse(data, np.array([1, 0]), expdesign = d)
        p1[k,l] = e.calc_Fd()
        p2[k,l] = e.calc_Fe(ncond = 2)
        temp = e.roi/np.mean(e.roi)*100-100
        a1[k,l] = trapz(temp,dx =1)
        
        l += 1
    k += 1
    print(lisi)

        
#%%
from tools import plotfs
fig = plotfs.plotdata(p2,p1,200,70,normalize = True)
#fig.savefig("Figures/Singletrial/st_det_heff.png",dpi = 600,bbox_inches = 'tight')
#plotdata.plotdata(c1,c3,4500,4500)
#%%
from numpy.fft import fft
from tools._dghrf import _dghrf
x = np.concatenate(np.zeros(30),np.array(_dghrf()),np.zeros(30))
plt.plot(x)
plt.xlim(0,0.1)
        
        