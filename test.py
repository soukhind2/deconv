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
from sklearn.decomposition import PCA
import importlib as imp
import matplotlib.pyplot as plt
pca = PCA(n_components = 10,whiten = True)
p1 = np.zeros((18,18))
p2 = np.zeros((18,18))
a1 = np.zeros((18,18))
c1 = np.zeros((18,18))
c2 = np.zeros((18,18))
c3 = np.zeros((18,18))

k = 0

for lisi in np.arange(1,19,1):
    l = 0
    for uisi in np.arange(1,19,1):
        if lisi > uisi:
            l+=1
            continue
        d = design.expdesign(1, lisi, uisi, 0.1, 100, [102], lv)
        data = d.tcourse()
        e = design.expanalyse(data, np.array([1]), expdesign = d)
        p1[k,l] = e.calc_Fd()
        p2[k,l] = e.calc_Fe(ncond = 1)
        temp = e.roi/np.mean(e.roi)*100-100
        a1[k,l] = trapz(temp,dx =1)
        out = avgHRF(e.expdesign.onsets_A, temp, e.expdesign.loadvolume.tr) 
        temp = temp.reshape(1,-1)
        pca.fit(out)
        c1[k,l] = pca.explained_variance_[0]
        c2[k,l] = pca.explained_variance_[1]
        c3[k,l] = pca.explained_variance_[2]
        if lisi == 15 and uisi == 17:
            exroi1 = temp
            print("1")
        if lisi == 1 and uisi == 7:
            exroi2 = temp
            print("2")
        if lisi == 7 and uisi == 18:
            exroi3 = temp
            print("3")
        l += 1
    k += 1
    print(lisi)

        
#%%
from tools import plotdata
fig = plotdata.plotdata(c1,p2,200,70,normalize = True)
#fig.savefig("Figures/Singletrial/st_det_heff.png",dpi = 600,bbox_inches = 'tight')
#plotdata.plotdata(c1,c3,4500,4500)

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
for i in range(1,5):
    for j in range(1,5):
        if i>j:
            continue
        print(i,j)
        
        
        
        
        