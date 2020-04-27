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
cue_ratio = np.array([1,0.9,0.8,0.7,0.6,0.5]) 
#%matplotlib qt
uisi = np.arange(2,21,2)
lisi = 2
c1 = np.zeros((len(cue_ratio),len(uisi)))
c2 = np.zeros((len(cue_ratio),len(uisi)))
sumc1 = np.zeros((10,len(cue_ratio),len(uisi)))
sumc2 = np.zeros((10,len(cue_ratio),len(uisi)))


for count in range(0,10):
    c1 = np.zeros((len(cue_ratio),len(uisi)))
    c2 = np.zeros((len(cue_ratio),len(uisi)))
    k = 0
    for ratio in cue_ratio:
        l = 0
        for y in uisi:
            if lisi > y:
                continue
            lower_isi = lisi
            upper_isi = y
            d = design.expdesign(ratio, lisi, y, 0.1, 100, [2], lv)
            data = d.tcourse()
            e = design.expanalyse(data, np.array([1, 0]), expdesign = d)
            c1[k,l] = e.calc_Fd()
            c2[k,l] = e.calc_Fe(ncond = 2)
            l = l + 1
        
        k += 1
    sumc1[count] = c1
    sumc2[count] = c2
    print(count)

        
#%%
avgc1 = np.mean(sumc1,0)
avgc2 = np.mean(sumc2,0)
avgc1.shape

#%%
fig= plt.figure()
ax = fig.add_subplot(111)
plt.plot(avgc1[0],'-D',label = '0',c = 'red')
plt.plot(avgc1[1],'-o',label = '0.1',c = 'black')
plt.plot(avgc1[2],'-o',label = '0.2',c = 'orange')
plt.plot(avgc1[3],'-x',label = '0.3',c = 'lime')
plt.plot(avgc1[4],'-h',label = '0.4',c = 'violet')
plt.plot(avgc1[5],'-*',label = '0.5',c = 'blue')
plt.legend(loc = 4,title = 'Proportion')
#ax.set_xlabel('Upper Bound of SOA')
ax.set_ylabel('Detection Power')
#ax.set_ylabel('HRF Estimation Eff.')
#ax.set_title('HRF Estimation Eff. for Different Cue Only Trial Proportions(lsoa = 4)')
ax.set_title('Detection Power for Different Cue Only Trial Proportions(lsoa = 2)')
labels = uisi.astype(str)
labels[0] = '2s Const. ISI'
plt.xticks(np.arange(0,len(uisi),1), labels)
#plt.scatter(avgc1[0],avgc2[0])
        