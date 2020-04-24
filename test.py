
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

# %%

import design
import numpy as np
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import importlib as imp
import matplotlib.pyplot as plt
pca = PCA(n_components = 10)
p1 = np.zeros((38,38))

p2 = np.zeros((38,38))
a1 = np.zeros((38,38))
c1 = np.zeros((38,38))
c2 = np.zeros((38,38))
c3 = np.zeros((38,38))
k =0
l = 0
for lisi in np.arange(1,19,0.5):
    for uisi in np.arange(1,19,0.5):
        if lisi > uisi:
            continue
        d = design.expdesign(1, lisi, uisi, 0.1, 100, [102], lv)
        data = d.tcourse()
        e = design.expanalyse(data, np.array([1]), expdesign = d)
        p1[k,l] = e.calc_Fd()
        p2[k,l] = e.calc_Fe(ncond = 1)
        m_data = e.roi/np.mean(e.roi)*100-100
        a1[k,l] = trapz(m_data,dx = 1)
        out = avgHRF(e.expdesign.onsets_A, m_data, e.expdesign.loadvolume.tr)
        pca.fit(out)
        c1[k,l] = pca.explained_variance_[0]
        c2[k,l] = pca.explained_variance_[1]
        c3[k,l] = pca.explained_variance_[2]
        if uisi == 14 and lisi == 2:
            exroi1 = e.roi
            out1 = avgHRF(e.expdesign.onsets_A, e.roi, e.expdesign.loadvolume.tr)
            onsetA1 = d.onsets_A
            print('regis 1')
        elif uisi == 17 and lisi == 4:
            exroi2 = e.roi
            out2 = avgHRF(e.expdesign.onsets_A, e.roi, e.expdesign.loadvolume.tr)
            onsetA2 = d.onsets_A
            print('regis 2')
        elif uisi == 17 and lisi == 7:
            exroi3 = e.roi
            out3 = avgHRF(e.expdesign.onsets_A, e.roi, e.expdesign.loadvolume.tr)
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
            l = l +1
        k = k+1
    print(lisi)

        

        
        
        
        
        
        