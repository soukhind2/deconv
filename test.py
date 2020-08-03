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
            d = design.expdesign(ratio, 1,lisi, y, 0.1, 100, [2], lv,noise = True,
                                 nonlinear = True,
                                 load = 'wmmap')
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
fig= plt.figure(figsize = (12,8))
ax = fig.add_subplot(111)
plt.plot(avgc2[0],'-D',label = '0',c = 'b')
plt.plot(avgc2[1],'-o',label = '10%',c = 'g')
plt.plot(avgc2[2],'-^',label = '20%',c = 'r')
plt.plot(avgc2[3],'-x',label = '30%',c = 'c')   
plt.plot(avgc2[4],'-s',label = '40%',c = 'm')
plt.plot(avgc2[5],'-*',label = '50%',c = 'y')
l = plt.legend(loc = 4,title = 'Proportion of \n partial trials')
plt.setp(l.get_title(), multialignment='center')
#ax.set_ylabel('Detection Power',size = 15)
ax.set_xlabel('Upper Bound of ISI',size = 15)
ax.set_ylabel('Estimation Eff.',size = 15)
ax.set_title('Estimation Eff. for Partial - Trial Proportions')
#ax.set_title('Detection Power for Partial - Trial Proportions',size = 15)
labels = uisi.astype(str)
labels[0] = '2s Const.'
plt.xticks(np.arange(0,len(uisi),1), labels,size = 15)
plt.yticks(size = 15)
#plt.scatter(avgc1[0],avgc2[0])
#fig.savefig("Figures/Nullevents/N2.png",bbox_inches = "tight",dpi = 600)
        