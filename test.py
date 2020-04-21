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
pca = PCA(n_components = 10)
p1 = np.zeros((19,19))

p2 = np.zeros((19,19))
a1 = np.zeros((19,19))
c1 = np.zeros((19,19))
c2 = np.zeros((19,19))
c3 = np.zeros((19,19))

for lisi in range(1,19):
    for uisi in range(1,19):
        if lisi > uisi:
            continue
        d = design.expdesign(1, lisi, uisi, 0.1, 100, [0.05], lv)
        data = d.tcourse()
        e = design.expanalyse(data, np.array([1]), expdesign = d)
        p1[lisi-1,uisi-1] = e.calc_Fd()
        p2[lisi-1,uisi-1] = e.calc_Fe(ncond = 1)
        a1[lisi-1,uisi-1] = trapz(e.roi,dx =1)
        out = avgHRF(e.expdesign.onsets_A, e.roi, e.expdesign.loadvolume.tr)
        pca.fit(out)
        c1[lisi-1,uisi-1] = pca.explained_variance_[0]
        c2[lisi-1,uisi-1] = pca.explained_variance_[1]
        c3[lisi-1,uisi-1] = pca.explained_variance_[2]
        if uisi == 14 and lisi == 2:
            exroi1 = e.roi
            onsetA1 = d.onsets_A
            print('regis 1')
        elif uisi == 17 and lisi == 4:
            exroi2 = e.roi
            onsetA2 = d.onsets_A
            print('regis 2')
        elif uisi == 17 and lisi == 7:
            exroi3 = e.roi
            onsetA3 = d.onsets_A
            print('regis 3')
        elif uisi == 8 and lisi == 5:
            exroi4 = e.roi
            onsetA4 = d.onsets_A
            print('regis 4')
        elif uisi == 8 and lisi == 2:
            exroi5 = e.roi
            onsetA5 = d.onsets_A
            print('regis 5')  
        elif uisi == 19 and lisi == 11:
            exroi6 = e.roi
            onsetA6 = d.onsets_A
            print('regis 6')
        elif uisi == 3 and lisi == 2:
            exroi7 = e.roi
            onsetA7 = d.onsets_A
            print('regis 7')
    print(lisi)

        
#%%
from tools import plotdata
plotdata.plotdata(p1,p2,2500,3)
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

out = avgHRF(e.expdesign.onsets_A, e.roi, e.expdesign.loadvolume.tr)
#%%
onsets = e.expdesign.onsets_A
brain = e.roi
tr = e.expdesign.loadvolume.tr
res = 20
out = np.zeros((len(onsets),res))
for i in range(0,len(onsets)):
    start = (onsets[i] / tr).astype('int')
    series_A = np.linspace(start - 3,start + 16,res).astype('int')
    if series_A[-1]  >= brain.shape[0]:
        break
    else:
        out[i,:] = brain[series_A]

        
        
        
        
        
        
        
        