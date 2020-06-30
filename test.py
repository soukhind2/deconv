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
        d = design.expdesign(1, lisi, uisi, 0.1, 100, [102], lv,noise = True)
        data = d.tcourse()
        e = design.expanalyse(data, np.array([1]), expdesign = d)
        p1[k,l] = e.calc_Fd()
        p2[k,l] = e.calc_Fe(ncond = 1)
        temp = e.roi/np.mean(e.roi)*100-100
        a1[k,l] = trapz(temp,dx =1)
        
        l += 1
    k += 1
    print(lisi)

        
#%%
from tools import plotfs
fig = plotfs.plotdata(p1,p2,200,70,normalize = True)
#fig.savefig("Figures/Singletrial/st_det_heff.png",dpi = 600,bbox_inches = 'tight')
#plotdata.plotdata(c1,c3,4500,4500)

        
#%%
plt.plot(d.stimfunc_weighted*600)
#%%

fig= plt.figure()
d = design.expdesign(1, 2, 3, 0.1, 100, [102], lv , noise = True)
data = d.tcourse()  
e = design.expanalyse(data, np.array([1]), expdesign = d)
plt.plot(e.roi)
#%%
fig= plt.figure()
plt.plot(e.boxcar_A)
plt.plot(d.temp)
#%%
test = np.zeros((20,20))
for l in np.arange(1,20,1):
    for u in np.arange(1,20,1):

        if l <= 7 and u <= 9:
            test[l,u] = test[l,u] + 1
            
        elif ((l <= 7 and u >= 10 and u <= 13) or (l <= 5 and u >= 14)
              or (l == 8 and u >= 8 and u <= 9)
              or (l == 9 and u == 9)):
            test[l,u] = test[l,u] + 2
            
        elif (((l == 6 or l == 7) and u >= 14 and u <= 20) 
              or ((l >=8 and l <= 10) and u >= 10 and u <= 13)
              or ((l >= 10 and l <= 13) and u >= 11 and u <= 13)):
            test[l,u] = test[l,u] + 3
            
        elif (((l == 8 or l == 9) and u >= 14 and u <= 20) 
            or ((l == 10 or l == 11) and u >= 14 and u <= 20)
            or ((l >= 12 and l <= 14) and u >= 14 and u <= 20)
            or ((l == 14 or l == 15) and u >= 15 and u <= 20)
            or (l >= 16 and u >= 15 and u <= 20)):
            test[l,u] = test[l,u] + 4


        


